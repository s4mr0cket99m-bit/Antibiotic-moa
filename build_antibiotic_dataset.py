import json
import os
import sys
import math
import argparse
import subprocess
import logging
from typing import List, Dict, Optional, Tuple, Any

import numpy as np

# Optional imports guarded to allow partial execution when dependencies are missing
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolAlign
    from rdkit.Chem.rdchem import Mol
except ImportError:  # pragma: no cover
    Chem = None
    AllChem = None
    rdMolAlign = None
    Mol = None

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

try:  # pragma: no cover
    from pyscf import gto, dft
except ImportError:
    gto = None
    dft = None


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# Grid parameters tuned for downstream CNN/Hailo: 48^3 grid with 0.5 Å spacing (~24 Å box)
GRID_SIZE = 48
GRID_SPACING = 0.5
PADDING = 4.0  # Å padding around molecule when deriving bounding boxes


# ---------------------------
# Task 1: Molecule lists
# ---------------------------

FALLBACK_SMILES: Dict[str, str] = {
    # Beta-lactams
    "penicillin G": "CC1(C)SC2C(NC(=O)C(C3=CC=CC=C3)N2C(=O)O1)C(=O)O",
    "amoxicillin": "CC1(C)SC2C(NC(=O)C(N)C(N)N2C(=O)O1)C(=O)O",
    "ampicillin": "CC1(C)SC2C(NC(=O)C(N)C(N)N2C(=O)O1)C(=O)O",  # simplified
    "oxacillin": "CC1(C)SC2C(NC(=O)C(C3=CC=CC=C3O)N2C(=O)O1)C(=O)O",
    "cefazolin": r"CC1=C(NC(=O)C(N2C(=O)SC3C2SC(=C(N3)CO)CO)C(=O)O)CS/C(=N/OC)/N\O1",
    "ceftriaxone": r"CC1=NN(C(=O)C2=C(N3C(=O)SC4C3SC(=C(N4)CO)CO)C(=O)O)C(=N/O1)CS(=O)(=O)OCCN",
    "cefepime": r"CC1=NN(C(=O)C2=C(N3C(=O)SC4C3SC(=C(N4)CO)CO)C(=O)O)C(=N/O1)CS(=O)(=O)N(CC)CC",
    "imipenem": "CC1C(N2C1=O)C(=O)N[C@@H](C(C)=O)C2CCO",
    "meropenem": "CC1C(N2C1=O)C(=O)N[C@@H](C(C)=O)C2CCOC(=O)NCCN",
    "aztreonam": "CC1=NN(C(=O)C2=C(N3C(=O)SC(=N)C3)C(=O)O)C(=N/O1)CS(=O)(=O)NC",
    # Non beta-lactam PBP/cell-wall inhibitors
    "lactivicin": "CC1=CC(=O)N2C1CCOC2=O",
    "oxadiazole": "COC1=NC(=NO1)C2=CC=CC=C2",  # representative PBP2a inhibitor scaffold
    "fosfomycin": "OCC1OC(O)O1",
    "D-cycloserine": "C1C(NC(=O)O1)O",
    "bacitracin": "CC(C)C[C@H]1C(=O)N[C@H](C(=O)N[C@H](C(=O)N[C@H](C(=O)N[C@H](C(=O)N)CO)CO)CO)NC(=O)[C@H](CC2=CC=CC=C2)NC(=O)[C@H](NC1=O)CO",
    "vancomycin": "CC1=C(C(=O)N)C=CC=C1O",  # simplified surrogate
}


def query_pubchem_smiles(name: str) -> Optional[str]:
    if requests is None:
        return None
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{requests.utils.quote(name)}/property/CanonicalSMILES/JSON"
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json()
        return data["PropertyTable"]["Properties"][0]["CanonicalSMILES"]
    except Exception:
        return None


def query_smiles(name: str) -> Optional[str]:
    # Try API first, then fallback dictionary
    smiles = query_pubchem_smiles(name)
    if smiles:
        return smiles
    return FALLBACK_SMILES.get(name)


def build_compound_records() -> List[Dict[str, Any]]:
    beta_lactam_names = [
        "penicillin G",
        "amoxicillin",
        "ampicillin",
        "oxacillin",
        "cefazolin",
        "ceftriaxone",
        "cefepime",
        "imipenem",
        "meropenem",
        "aztreonam",
    ]
    non_beta = [
        "lactivicin",
        "oxadiazole",
        "fosfomycin",
        "D-cycloserine",
        "bacitracin",
        "vancomycin",
    ]
    records: List[Dict[str, Any]] = []
    for nm in beta_lactam_names:
        smi = query_smiles(nm)
        if smi:
            records.append({"name": nm, "smiles": smi, "class_label": 1})
        else:
            logging.warning(f"No SMILES found for {nm}")
    for nm in non_beta:
        smi = query_smiles(nm)
        if smi:
            records.append({"name": nm, "smiles": smi, "class_label": 0})
        else:
            logging.warning(f"No SMILES found for {nm}")
    return records


def save_records(records: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(records, f, indent=2)
    logging.info(f"Saved molecule list to {path}")


# ---------------------------
# Task 2: 3D structure generation and alignment
# ---------------------------


def ensure_rdkit() -> None:
    if Chem is None:
        raise ImportError("RDKit is required for 3D generation; please install rdkit-pypi.")


def generate_3d_mol(smiles: str) -> Optional[Mol]:
    ensure_rdkit()
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        AllChem.EmbedMolecule(mol, params)
        # Optimize geometry with MMFF94, fallback to UFF
        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMolecule(mol)
        else:
            AllChem.UFFOptimizeMolecule(mol)
        return mol
    except Exception as e:
        logging.error(f"3D generation failed for {smiles}: {e}")
        return None


# SMARTS for beta-lactam ring: four-membered ring with carbonyl C, amide N
BETA_LACTAM_SMARTS = "[#6]1(=O)[#7][#6][#6]1"


def align_beta_lactam(mol: Mol, ref_mol: Mol, ref_match: Tuple[int, ...]) -> Mol:
    # Align beta-lactam ring to reference using atom mapping
    matches = mol.GetSubstructMatches(Chem.MolFromSmarts(BETA_LACTAM_SMARTS))
    if not matches:
        return align_by_principal_axes(mol)
    match = matches[0]
    mapping = list(zip(match, ref_match))
    rdMolAlign.AlignMol(mol, ref_mol, atomMap=mapping)
    center_at_origin(mol)
    return mol


def align_by_principal_axes(mol: Mol) -> Mol:
    # Center at origin and rotate to principal axes for reproducibility
    conf = mol.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    centroid = coords.mean(axis=0)
    coords_centered = coords - centroid
    cov = np.cov(coords_centered.T)
    vals, vecs = np.linalg.eigh(cov)
    rot = vecs
    rotated = coords_centered @ rot
    for i, pt in enumerate(rotated):
        conf.SetAtomPosition(i, pt.tolist())
    return mol


def center_at_origin(mol: Mol) -> None:
    conf = mol.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    centroid = coords.mean(axis=0)
    for i, pt in enumerate(coords - centroid):
        conf.SetAtomPosition(i, pt.tolist())


def build_reference_beta_lactam() -> Tuple[Mol, Tuple[int, ...]]:
    ensure_rdkit()
    ref = generate_3d_mol("O=C1CNC1")  # azetidin-2-one
    if ref is None:
        raise RuntimeError("Failed to build reference beta-lactam scaffold")
    match = ref.GetSubstructMatches(Chem.MolFromSmarts(BETA_LACTAM_SMARTS))
    if not match:
        raise RuntimeError("Reference beta-lactam SMARTS did not match")
    center_at_origin(ref)
    return ref, match[0]


def save_sdf(mol: Mol, path: str) -> None:
    writer = Chem.SDWriter(path)
    writer.write(mol)
    writer.close()


def optimize_with_openbabel(mol_path: str) -> None:
    # Optional additional minimization using Open Babel if available
    try:
        subprocess.run(["obminimize", "-ff", "GAFF", "-n", "200", mol_path], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        logging.debug("Open Babel obminimize not found; skipping")


# ---------------------------
# Task 3: Electron density representations
# ---------------------------


def build_grid(grid_size: int = GRID_SIZE, spacing: float = GRID_SPACING) -> Tuple[np.ndarray, np.ndarray]:
    # Returns meshgrid and coordinate list
    half = grid_size // 2
    axes = (np.arange(grid_size) - half) * spacing
    X, Y, Z = np.meshgrid(axes, axes, axes, indexing='ij')
    coords = np.stack([X, Y, Z], axis=-1)
    flat_coords = coords.reshape(-1, 3)
    return coords, flat_coords


def mol_to_pyscf(mol: Mol) -> Optional[Any]:
    if gto is None:
        return None
    conf = mol.GetConformer()
    atoms = []
    for i, atom in enumerate(mol.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        atoms.append((atom.GetSymbol(), (pos.x, pos.y, pos.z)))
    pmol = gto.Mole()
    pmol.atom = atoms
    pmol.basis = '6-31g*'
    pmol.build()
    return pmol


def compute_dft_density(mol: Mol, grid_coords: np.ndarray) -> Optional[np.ndarray]:
    # grid_coords: (N,3)
    if dft is None or gto is None:
        logging.warning("PySCF not installed; skipping DFT density")
        return None
    try:
        pmol = mol_to_pyscf(mol)
        if pmol is None:
            return None
        mf = dft.RKS(pmol)
        mf.xc = 'b3lyp'
        mf.kernel()
        dm = mf.make_rdm1()
        ao = mf._numint.eval_ao(pmol, grid_coords)
        rho = mf._numint.eval_rho(pmol, ao, dm)
        return np.asarray(rho, dtype=np.float32)
    except Exception as e:
        logging.error(f"DFT density failed: {e}")
        return None


def gaussian_density(mol: Mol, grid_coords: np.ndarray, spacing: float = GRID_SPACING) -> np.ndarray:
    # Simple atom-centered Gaussian density as lightweight proxy
    conf = mol.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    Z = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=np.float32)
    sigma = 0.6  # Å width
    density = np.zeros(len(grid_coords), dtype=np.float32)
    for c, z in zip(coords, Z):
        diff = grid_coords - c
        dist2 = np.sum(diff * diff, axis=1)
        density += z * np.exp(-dist2 / (2 * sigma ** 2))
    return density


def huckel_density(mol: Mol, grid_coords: np.ndarray) -> np.ndarray:
    # Crude Huckel-like pi electron smear: aromatic atoms contribute more
    conf = mol.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    density = np.zeros(len(grid_coords), dtype=np.float32)
    sigma = 0.8
    for atom, c in zip(mol.GetAtoms(), coords):
        weight = 1.0
        if atom.GetIsAromatic():
            weight = 2.0
        elif atom.GetAtomicNum() in (7, 8):
            weight = 1.5
        diff = grid_coords - c
        dist2 = np.sum(diff * diff, axis=1)
        density += weight * np.exp(-dist2 / (2 * sigma ** 2))
    return density


# ---------------------------
# Task 4: Metadata tensors
# ---------------------------


def periodic_group(atomic_num: int) -> int:
    # Simple mapping for main elements encountered
    groups = {
        1: 1, 6: 14, 7: 15, 8: 16, 9: 17,
        15: 15, 16: 16, 17: 17, 35: 17, 53: 17,
        11: 1, 12: 2, 19: 1, 20: 2, 30: 12
    }
    return groups.get(atomic_num, 0)


def element_and_group_grids(mol: Mol, grid_shape: Tuple[int, int, int], spacing: float = GRID_SPACING) -> Tuple[np.ndarray, np.ndarray]:
    elem_grid = np.zeros(grid_shape, dtype=np.int16)
    group_grid = np.zeros(grid_shape, dtype=np.int16)
    conf = mol.GetConformer()
    half = grid_shape[0] // 2
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        idx = np.round(np.array([pos.x, pos.y, pos.z]) / spacing).astype(int) + half
        if np.all((idx >= 0) & (idx < grid_shape[0])):
            elem_grid[tuple(idx)] = atom.GetAtomicNum()
            group_grid[tuple(idx)] = periodic_group(atom.GetAtomicNum())
    return elem_grid, group_grid


def coordinate_grid(grid_shape: Tuple[int, int, int], spacing: float = GRID_SPACING) -> np.ndarray:
    half = grid_shape[0] // 2
    axes = (np.arange(grid_shape[0]) - half) * spacing
    X, Y, Z = np.meshgrid(axes, axes, axes, indexing='ij')
    coords = np.stack([X, Y, Z], axis=-1).astype(np.float32)
    return coords


def bond_adjacency(mol: Mol) -> np.ndarray:
    n = mol.GetNumAtoms()
    adj = np.zeros((n, n), dtype=np.int8)
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        btype = bond.GetBondType()
        order = 1
        if btype == Chem.BondType.DOUBLE:
            order = 2
        elif btype == Chem.BondType.TRIPLE:
            order = 3
        elif btype == Chem.BondType.AROMATIC:
            order = 4
        adj[i, j] = adj[j, i] = order
    return adj


def bond_mask_grid(mol: Mol, grid_shape: Tuple[int, int, int], spacing: float = GRID_SPACING) -> np.ndarray:
    mask = np.zeros(grid_shape, dtype=np.int8)
    conf = mol.GetConformer()
    half = grid_shape[0] // 2
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        pi = np.array(conf.GetAtomPosition(i))
        pj = np.array(conf.GetAtomPosition(j))
        for t in np.linspace(0, 1, num=10):
            p = pi * (1 - t) + pj * t
            idx = np.round(p / spacing).astype(int) + half
            if np.all((idx >= 0) & (idx < grid_shape[0])):
                mask[tuple(idx)] = 1
    return mask


# ---------------------------
# Pipeline
# ---------------------------


def process_molecule(rec: Dict[str, Any], ref: Optional[Mol], ref_match: Optional[Tuple[int, ...]], output_dir: str, skip_dft: bool = False) -> Optional[Dict[str, Any]]:
    smiles = rec['smiles']
    name = rec['name']
    class_label = rec['class_label']
    mol = generate_3d_mol(smiles)
    if mol is None:
        return None
    # Alignment
    if class_label == 1 and ref is not None and ref_match is not None:
        mol = align_beta_lactam(mol, ref, ref_match)
    else:
        mol = align_by_principal_axes(mol)
    # Save aligned structure
    sdf_path = os.path.join(output_dir, 'aligned_structures', f"{name.replace(' ', '_')}.sdf")
    os.makedirs(os.path.dirname(sdf_path), exist_ok=True)
    save_sdf(mol, sdf_path)
    optimize_with_openbabel(sdf_path)

    grid, flat_coords = build_grid()
    # Electron densities
    density_dft = None if skip_dft else compute_dft_density(mol, flat_coords)
    density_gauss = gaussian_density(mol, flat_coords)
    density_huckel = huckel_density(mol, flat_coords)

    elem_grid, group_grid = element_and_group_grids(mol, grid.shape[:3])
    coord_grid = coordinate_grid(grid.shape[:3])
    bonds = bond_adjacency(mol)
    bond_mask = bond_mask_grid(mol, grid.shape[:3])

    # Reshape densities to grid form
    def reshape_or_none(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if arr is None:
            return None
        return arr.reshape(grid.shape[:3])

    record = {
        "name": name,
        "class_label": class_label,
        "density_dft": reshape_or_none(density_dft),
        "density_gaussian": reshape_or_none(density_gauss),
        "density_huckel": reshape_or_none(density_huckel),
        "element_grid": elem_grid,
        "group_grid": group_grid,
        "coord_grid": coord_grid,
        "bond_adj": bonds,
        "bond_mask": bond_mask,
        "smiles": smiles,
    }
    # Save per-molecule npz for convenience
    mol_npz = os.path.join(output_dir, 'npz', f"{name.replace(' ', '_')}.npz")
    os.makedirs(os.path.dirname(mol_npz), exist_ok=True)
    np.savez_compressed(mol_npz, **{k: v for k, v in record.items() if v is not None})
    return record


def build_dataset(output_dir: str, max_molecules: Optional[int] = None, skip_dft: bool = False) -> List[Dict[str, Any]]:
    records = build_compound_records()
    save_records(records, os.path.join(output_dir, 'antibiotic_smiles.json'))
    ensure_rdkit()
    ref, ref_match = build_reference_beta_lactam()
    dataset = []
    for i, rec in enumerate(records):
        if max_molecules is not None and i >= max_molecules:
            break
        logging.info(f"Processing {rec['name']}")
        processed = process_molecule(rec, ref, ref_match, output_dir, skip_dft=skip_dft)
        if processed:
            dataset.append(processed)
    if dataset:
        os.makedirs(output_dir, exist_ok=True)
        np.savez_compressed(os.path.join(output_dir, 'pbp_antibiotics_tensors.npz'), data=np.array(dataset, dtype=object))
        logging.info(f"Saved aggregated dataset with {len(dataset)} molecules")
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Build antibiotic electron-density dataset")
    parser.add_argument("--output-dir", default="data", help="Output directory for JSON, SDF, and NPZ files")
    parser.add_argument("--max-molecules", type=int, default=None, help="Limit number of molecules for quick runs")
    parser.add_argument("--skip-dft", action="store_true", help="Skip DFT calculation and use only proxy densities")
    args = parser.parse_args()
    build_dataset(args.output_dir, max_molecules=args.max_molecules, skip_dft=args.skip_dft)


if __name__ == "__main__":
    main()
