from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from .models import Antibiotic, Atom


def load_antibiotics(json_path: str | Path) -> List[Antibiotic]:
    path = Path(json_path)
    content = json.loads(path.read_text())
    antibiotics: List[Antibiotic] = []
    for entry in content:
        atoms = [
            Atom(
                label=atom["label"],
                atomic_number=int(atom["atomic_number"]),
                position=tuple(atom["position"]),
                partial_charge=float(atom.get("partial_charge", 0.0)),
                properties={k: float(v) for k, v in atom.get("properties", {}).items()},
            )
            for atom in entry["atoms"]
        ]
        antibiotics.append(
            Antibiotic(
                name=entry["name"],
                mechanism=entry["mechanism"],
                atoms=atoms,
            )
        )
    return antibiotics


def save_tensor(path: Path, tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(tensor, fh)


def save_molecule_tensors(output_dir: Path, antibiotic: Antibiotic, tensors) -> None:
    save_tensor(output_dir / f"{antibiotic.name}_position.json", tensors.position_matrix)
    save_tensor(output_dir / f"{antibiotic.name}_type.json", tensors.type_matrix)
    save_tensor(output_dir / f"{antibiotic.name}_property.json", tensors.property_matrix)
    save_tensor(output_dir / f"{antibiotic.name}_density.json", tensors.density_matrix)


def save_super_matrices(output_dir: Path, label: str, tensors) -> None:
    save_tensor(output_dir / f"super_{label}_position.json", tensors.position_matrix)
    save_tensor(output_dir / f"super_{label}_type.json", tensors.type_matrix)
    save_tensor(output_dir / f"super_{label}_property.json", tensors.property_matrix)
    save_tensor(output_dir / f"super_{label}_density.json", tensors.density_matrix)

