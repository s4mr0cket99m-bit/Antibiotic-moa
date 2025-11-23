from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class Atom:
    """Lightweight container for atomic information.

    Positions are recorded in angstroms relative to an origin defined by the
    largest atom in the molecule's backbone. The atomic number encodes the
    element and can be used for mapping to electron configuration data.
    """

    label: str
    atomic_number: int
    position: Tuple[float, float, float]
    partial_charge: float
    properties: Dict[str, float] = field(default_factory=dict)


@dataclass
class Antibiotic:
    """Definition for an antibiotic small molecule drug (SMD)."""

    name: str
    mechanism: str
    atoms: List[Atom]


@dataclass
class MoleculeTensors:
    """All tensor representations produced for an antibiotic molecule."""

    position_matrix: List[Tuple[float, float, float]]
    type_matrix: List[List[List[float]]]
    property_matrix: List[List[List[float]]]
    density_matrix: List[List[List[float]]]

