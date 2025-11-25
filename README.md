# Antibiotic-moa

This repository builds a compact antibiotic dataset with 3D electron-density proxies and aligned metadata tensors suitable for CNN-based models and accelerator deployment.

## Requirements
- Python 3.9+
- NumPy
- RDKit (`pip install rdkit-pypi`)
- PySCF (`pip install pyscf`) for DFT densities (optional; can be skipped with `--skip-dft`)
- requests (optional, for PubChem lookup)
- Open Babel's `obminimize` (optional extra geometry refinement)

## Usage
Run the pipeline (may take time for quantum calculations):
```bash
python build_antibiotic_dataset.py --output-dir data
```
For faster runs without DFT densities:
```bash
python build_antibiotic_dataset.py --output-dir data --skip-dft
```
Outputs include `data/antibiotic_smiles.json`, aligned SDF files under `data/aligned_structures/`, per-molecule `.npz` tensors, and an aggregated `pbp_antibiotics_tensors.npz`.
