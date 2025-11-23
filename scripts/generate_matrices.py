#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from antibiotic_moa.io import load_antibiotics, save_molecule_tensors, save_super_matrices
from antibiotic_moa.tensors import aggregate_super_matrices, build_molecule_tensors


def generate_for_antibiotics(
    antibiotics_path: Path, output_dir: Path, spacing: float, margin: float, xy_resolution: int
) -> None:
    antibiotics = load_antibiotics(antibiotics_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    tensors_by_mechanism: Dict[str, List] = defaultdict(list)

    for antibiotic in antibiotics:
        tensors = build_molecule_tensors(
            antibiotic, spacing=spacing, margin=margin, xy_resolution=xy_resolution
        )
        save_molecule_tensors(output_dir, antibiotic, tensors)
        tensors_by_mechanism[antibiotic.mechanism].append(tensors)

    all_tensors = [t for tensors in tensors_by_mechanism.values() for t in tensors]
    global_super = aggregate_super_matrices(all_tensors)
    save_super_matrices(output_dir, "all", global_super)

    for mechanism, tensors in tensors_by_mechanism.items():
        super_tensor = aggregate_super_matrices(tensors)
        save_super_matrices(output_dir, mechanism, super_tensor)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate tensor matrices for antibiotic SMDs")
    parser.add_argument("--input", type=Path, default=Path("data/antibiotics.json"), help="Path to antibiotic definitions JSON")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="Directory to store tensor JSON files")
    parser.add_argument("--spacing", type=float, default=0.5, help="Grid spacing in angstroms for the tensor grid")
    parser.add_argument("--margin", type=float, default=2.0, help="Margin to pad around atom coordinates in angstroms")
    parser.add_argument(
        "--xy-resolution",
        type=int,
        default=1024,
        help="Target XY resolution for all tensor matrices (nearest-neighbor upsampled)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_for_antibiotics(
        args.input, args.output, spacing=args.spacing, margin=args.margin, xy_resolution=args.xy_resolution
    )


if __name__ == "__main__":
    main()

