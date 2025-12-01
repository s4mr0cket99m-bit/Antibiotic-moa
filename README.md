# Antibiotic-moa

This repository provides a placeholder workflow for iterative antibiotic scaffold generation driven by electron-density tensors. Use `iterative_antibiotic_generator.py` to load beta-lactam training data, generate candidate densities, translate them to SMILES, and fine-tune a stub affinity model.

## Quick start

```bash
python iterative_antibiotic_generator.py \
    --binding-csv /path/to/beta_lactam_training_set.csv \
    --density-dir /path/to/grids_64 \
    --target-id CHEMBL2026 \
    --iterations 3
```

Outputs (metrics, generated densities, SMILES/conformer files) are written to the `outputs/` directory by default.
