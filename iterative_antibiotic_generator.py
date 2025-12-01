"""Iterative antibiotic scaffold generator.

This script orchestrates a placeholder end-to-end workflow for generative
antibiotic design driven by electron-density tensors and lightweight neural
network stubs. It demonstrates how to:

1. Load ligand electron-density tensors and binding data for beta-lactams.
2. Evaluate an affinity predictor on known antibiotics.
3. Generate new density backbones, translate them to SMILES, and validate them.
4. Regenerate densities for new scaffolds and re-predict affinities.
5. Iteratively fine-tune the affinity model with pseudo-labeled candidates.

Usage:
    python iterative_antibiotic_generator.py \
        --binding-csv /path/to/beta_lactam_training_set.csv \
        --density-dir /path/to/grids_64 \
        --target-id CHEMBL2026 \
        --iterations 3

All heavy computations (quantum chemistry, deep learning) are replaced with
stubs that mimic expected interfaces while keeping runtime minimal.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import shutil
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # Optional dependency for realistic SMILES handling.
    from rdkit import Chem
    from rdkit.Chem import AllChem

    RDKit_AVAILABLE = True
except Exception:  # pragma: no cover - if rdkit is absent we fall back to stubs.
    RDKit_AVAILABLE = False


Tensor = np.ndarray


@dataclass
class AffinityModel:
    """Placeholder affinity predictor.

    The model tracks a simple bias term derived from training data to mimic
    fine-tuning. Predictions combine random noise, target-specific offsets, and
    the density tensor mean to provide deterministic but variable outputs.
    """

    random_state: random.Random = field(default_factory=random.Random)
    bias: float = 0.0

    def predict_affinity(self, density_tensor: Tensor, target_id: str) -> float:
        target_hash = hash(target_id) % 1000 / 1000.0
        density_signal = float(np.mean(density_tensor))
        noise = self.random_state.normalvariate(0, 0.1)
        return self.bias + density_signal * 10 + target_hash + noise

    def fine_tune(self, training_data: Sequence[Tuple[Tensor, str, float]]) -> None:
        if not training_data:
            return
        observed = [label for *_rest, label in training_data]
        self.bias = float(np.mean(observed))

    def clone(self) -> "AffinityModel":
        clone_model = AffinityModel(random_state=self.random_state)
        clone_model.bias = self.bias
        return clone_model


@dataclass
class DensityGenerator:
    random_state: random.Random = field(default_factory=random.Random)

    def generate_density(
        self,
        target_id: str,
        reference_density: Optional[Tensor] = None,
        num_samples: int = 1,
    ) -> List[Tensor]:
        samples = []
        target_shift = hash(target_id) % 17 / 50.0
        for _ in range(num_samples):
            if reference_density is not None:
                base = reference_density.astype(np.float32)
                noise = self.random_state.normalvariate(0, 0.05)
                jitter = self.random_state.random() * 0.1
                sample = base + noise + jitter + target_shift
            else:
                sample = self.random_state.normalvariate(0, 0.2)
                sample = np.full((4, 64, 64, 64), sample, dtype=np.float32)
            samples.append(sample)
        return samples


@dataclass
class DensityToSmilesModel:
    random_state: random.Random = field(default_factory=random.Random)

    def density_to_smiles(
        self, density_tensor: Tensor, target_id: str, num_candidates: int = 5
    ) -> List[str]:
        base = abs(float(np.mean(density_tensor)))
        candidates = []
        for i in range(num_candidates):
            token = self.random_state.randint(1, 9999)
            ring_count = 1 + int(base * 10) % 3
            smile = f"C1=CC=CC{ring_count}({target_id[:4]}{token})C=C1"
            candidates.append(smile)
        return candidates


def load_density_tensor(path: Path) -> Tensor:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    if path.suffix.lower() in {".h5", ".hdf5"}:  # pragma: no cover - optional path
        try:
            import h5py
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("h5py required to read HDF5 density files") from exc
        with h5py.File(path, "r") as handle:
            return np.array(handle["density"])
    raise ValueError(f"Unsupported density tensor format: {path}")


def list_density_files(density_dir: Path) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for ext in ("*.npy", "*.h5", "*.hdf5"):
        for path in density_dir.glob(ext):
            mapping[path.stem] = path
    return mapping


def read_binding_data(binding_csv: Path) -> List[Dict[str, str]]:
    with binding_csv.open() as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def associate_tensors(
    bindings: List[Dict[str, str]], tensor_map: Dict[str, Path]
) -> List[Tuple[Tensor, str, float, Dict[str, str]]]:
    paired = []
    missing = 0
    for row in bindings:
        chembl_id = row.get("molecule_chembl_id") or row.get("chembl_id")
        target_id = row.get("target_chembl_id") or ""
        affinity = row.get("pchembl_mean")
        if not chembl_id or affinity is None:
            continue
        tensor_path = tensor_map.get(chembl_id)
        if tensor_path is None:
            missing += 1
            continue
        tensor = load_density_tensor(tensor_path)
        paired.append((tensor, target_id, float(affinity), row))
    if missing:
        logging.warning("Skipped %d entries with missing density tensors", missing)
    return paired


def evaluate_model(
    model: AffinityModel, dataset: Sequence[Tuple[Tensor, str, float, Dict[str, str]]]
) -> Dict[str, float]:
    y_true: List[float] = []
    y_pred: List[float] = []
    for tensor, target, label, _meta in dataset:
        prediction = model.predict_affinity(tensor, target)
        y_true.append(label)
        y_pred.append(prediction)
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    mse = float(np.mean((y_true_arr - y_pred_arr) ** 2)) if len(y_true_arr) else math.inf
    mae = float(np.mean(np.abs(y_true_arr - y_pred_arr))) if len(y_true_arr) else math.inf
    corr = (
        float(np.corrcoef(y_true_arr, y_pred_arr)[0, 1])
        if len(y_true_arr) > 1
        else float("nan")
    )
    return {"mse": mse, "mae": mae, "corr": corr}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def validate_smiles(smiles: str) -> bool:
    if RDKit_AVAILABLE:
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=True)
            return mol is not None
        except Exception:
            return False
    return all(ch.isascii() for ch in smiles) and len(smiles) > 3


def generate_conformer(smiles: str, output_path: Path) -> None:
    if RDKit_AVAILABLE:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=0xf00d)
        AllChem.UFFOptimizeMolecule(mol, maxIters=50)
        Chem.MolToMolFile(mol, str(output_path))
    else:
        output_path.write_text(f"SMILES: {smiles}\n")


def compute_electron_density(smiles: str, seed: Optional[int] = None) -> Tensor:
    rng = np.random.default_rng(seed)
    base = rng.normal(loc=0.0, scale=0.2, size=(4, 64, 64, 64)).astype(np.float32)
    mod = (len(smiles) % 7) * 0.05
    return base + mod


def save_tensor(tensor: Tensor, path: Path) -> None:
    np.save(path, tensor)


def write_json(path: Path, payload: Dict) -> None:
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    keys = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def log_metrics(iteration: int, metrics: Dict[str, float]) -> None:
    logging.info(
        "Iteration %d metrics | MSE: %.4f | MAE: %.4f | Corr: %.4f",
        iteration,
        metrics.get("mse", float("nan")),
        metrics.get("mae", float("nan")),
        metrics.get("corr", float("nan")),
    )


def generate_candidates(
    generator: DensityGenerator,
    translator: DensityToSmilesModel,
    affinity_model: AffinityModel,
    target_id: str,
    output_dir: Path,
    reference_density: Optional[Tensor],
    num_density_samples: int,
    max_smiles_per_density: int,
) -> List[Dict[str, object]]:
    ensure_dir(output_dir)
    generated_records: List[Dict[str, object]] = []
    densities = generator.generate_density(target_id, reference_density, num_density_samples)
    for density_idx, density in enumerate(densities):
        density_id = f"density_{uuid.uuid4().hex[:8]}"
        density_path = output_dir / f"{density_id}.npy"
        save_tensor(density, density_path)
        smiles_candidates = translator.density_to_smiles(density, target_id, max_smiles_per_density)
        for smiles in smiles_candidates:
            if not validate_smiles(smiles):
                continue
            conformer_path = output_dir / f"{density_id}_{uuid.uuid4().hex[:6]}.mol"
            generate_conformer(smiles, conformer_path)
            regenerated = compute_electron_density(smiles)
            regenerated_path = output_dir / f"{density_id}_regen.npy"
            save_tensor(regenerated, regenerated_path)
            predicted_affinity = affinity_model.predict_affinity(regenerated, target_id)
            generated_records.append(
                {
                    "density_id": density_id,
                    "density_path": str(density_path),
                    "regen_density_path": str(regenerated_path),
                    "smiles": smiles,
                    "conformer_path": str(conformer_path),
                    "predicted_affinity": predicted_affinity,
                    "target_id": target_id,
                }
            )
    return generated_records


def fine_tune_and_select(
    base_model: AffinityModel,
    base_dataset: Sequence[Tuple[Tensor, str, float, Dict[str, str]]],
    augmented_examples: Sequence[Tuple[Tensor, str, float]],
) -> Tuple[AffinityModel, Dict[str, float]]:
    tuned_model = base_model.clone()
    tuned_model.fine_tune(augmented_examples)
    metrics = evaluate_model(tuned_model, base_dataset)
    return tuned_model, metrics


def build_augmented_examples(
    generated_records: Sequence[Dict[str, object]],
) -> List[Tuple[Tensor, str, float]]:
    examples: List[Tuple[Tensor, str, float]] = []
    for record in generated_records:
        regen_path = Path(str(record["regen_density_path"]))
        if not regen_path.exists():
            continue
        density = load_density_tensor(regen_path)
        target = str(record["target_id"])
        affinity = float(record["predicted_affinity"])
        examples.append((density, target, affinity))
    return examples


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Iterative antibiotic generator")
    parser.add_argument("--binding-csv", required=True, type=Path)
    parser.add_argument("--density-dir", required=True, type=Path)
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--num-density-samples", type=int, default=2)
    parser.add_argument("--max-smiles-per-density", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    random_state = random.Random(args.seed)
    np.random.seed(args.seed)

    output_root = args.output_dir
    if output_root.exists():
        shutil.rmtree(output_root)
    ensure_dir(output_root)

    logging.info("Loading binding data from %s", args.binding_csv)
    binding_rows = read_binding_data(args.binding_csv)
    tensor_map = list_density_files(args.density_dir)
    paired_dataset = associate_tensors(binding_rows, tensor_map)
    if not paired_dataset:
        logging.error("No binding entries matched with density tensors. Exiting.")
        return 1

    affinity_model = AffinityModel(random_state=random_state)
    density_generator = DensityGenerator(random_state=random_state)
    translator = DensityToSmilesModel(random_state=random_state)

    baseline_metrics = evaluate_model(affinity_model, paired_dataset)
    log_metrics(0, baseline_metrics)
    write_json(output_root / "baseline_metrics.json", baseline_metrics)

    best_mae = baseline_metrics.get("mae", float("inf"))
    best_model = affinity_model

    reference_density = paired_dataset[0][0]

    for iteration in range(1, args.iterations + 1):
        iteration_dir = output_root / f"iteration_{iteration}"
        ensure_dir(iteration_dir)
        logging.info("Generating candidates for iteration %d", iteration)

        generated_records = generate_candidates(
            generator=density_generator,
            translator=translator,
            affinity_model=best_model,
            target_id=args.target_id,
            output_dir=iteration_dir,
            reference_density=reference_density,
            num_density_samples=args.num_density_samples,
            max_smiles_per_density=args.max_smiles_per_density,
        )
        write_csv(iteration_dir / "generated_records.csv", generated_records)

        augmented_examples = build_augmented_examples(generated_records)
        tuned_model, metrics = fine_tune_and_select(best_model, paired_dataset, augmented_examples)
        log_metrics(iteration, metrics)
        write_json(iteration_dir / "metrics.json", metrics)

        if metrics.get("mae", float("inf")) < best_mae:
            logging.info("Adopting improved model from iteration %d", iteration)
            best_mae = metrics["mae"]
            best_model = tuned_model
        else:
            logging.info("Iteration %d did not improve MAE (%.4f >= %.4f)", iteration, metrics.get("mae", float("inf")), best_mae)

    final_metrics = evaluate_model(best_model, paired_dataset)
    write_json(output_root / "final_metrics.json", final_metrics)
    log_metrics(args.iterations + 1, final_metrics)
    logging.info("Completed iterative generation. Outputs stored in %s", output_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
