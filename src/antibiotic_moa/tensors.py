from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np

from .models import Antibiotic, Atom, MoleculeTensors


def _grid_bounds(atoms: Iterable[Atom], spacing: float, margin: float) -> Tuple[np.ndarray, np.ndarray]:
    coords = np.array([atom.position for atom in atoms], dtype=float)
    min_corner = coords.min(axis=0) - margin
    max_corner = coords.max(axis=0) + margin
    grid_min = spacing * np.floor(min_corner / spacing)
    grid_max = spacing * np.ceil(max_corner / spacing)
    return grid_min, grid_max


def _grid_shape(grid_min: np.ndarray, grid_max: np.ndarray, spacing: float) -> Tuple[int, int, int]:
    lengths = grid_max - grid_min
    shape = np.floor(lengths / spacing).astype(int) + 1
    return int(shape[0]), int(shape[1]), int(shape[2])


def _map_to_index(position: Tuple[float, float, float], origin: np.ndarray, spacing: float) -> Tuple[int, int, int]:
    delta = np.array(position, dtype=float) - origin
    return tuple(np.round(delta / spacing).astype(int))


def _atomic_sigma(atomic_number: int) -> float:
    # Basic heuristic: scale variance by approximate covalent radii trends.
    base = 0.25
    if atomic_number <= 2:
        return base
    if atomic_number <= 10:
        return base * 1.5
    if atomic_number <= 18:
        return base * 2.0
    return base * 2.5


def _gaussian_density(grid: np.ndarray, center_idx: Tuple[int, int, int], sigma: float) -> np.ndarray:
    x, y, z = np.indices(grid.shape)
    cx, cy, cz = center_idx
    dist_sq = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
    exponent = -dist_sq / (2 * sigma**2)
    density = np.exp(exponent)
    density_sum = density.sum()
    if density_sum == 0:
        return density
    return density * (0.99 / density_sum)


def _combine_properties(atom: Atom, property_channels: List[str]) -> Dict[str, float]:
    values = {channel: 0.0 for channel in property_channels}
    for key, value in atom.properties.items():
        if key in values:
            values[key] = value
    values["partial_charge"] = atom.partial_charge
    return values


def _upsample_xy(array: np.ndarray, target_resolution: int) -> np.ndarray:
    """Upsample the first two axes to the requested resolution via nearest-neighbor repeat."""

    current_x, current_y = array.shape[:2]
    if current_x == target_resolution and current_y == target_resolution:
        return array

    scale_x = int(np.ceil(target_resolution / current_x))
    scale_y = int(np.ceil(target_resolution / current_y))
    repeated = np.repeat(np.repeat(array, scale_x, axis=0), scale_y, axis=1)
    return repeated[:target_resolution, :target_resolution, ...]


def build_molecule_tensors(
    antibiotic: Antibiotic,
    spacing: float = 0.5,
    margin: float = 2.0,
    xy_resolution: int = 1024,
    property_channels: Iterable[str] | None = None,
) -> MoleculeTensors:
    channels = list(property_channels) if property_channels else ["h_donor", "h_acceptor", "electrophile", "nucleophile"]

    grid_min, grid_max = _grid_bounds(antibiotic.atoms, spacing, margin)
    grid_shape = _grid_shape(grid_min, grid_max, spacing)
    origin = grid_min

    type_grid = np.zeros(grid_shape, dtype=float)
    property_grid = {channel: np.zeros(grid_shape, dtype=float) for channel in channels}
    property_grid["partial_charge"] = np.zeros(grid_shape, dtype=float)
    density_grid = np.zeros(grid_shape, dtype=float)

    for atom in antibiotic.atoms:
        idx = _map_to_index(atom.position, origin, spacing)
        type_grid[idx] = atom.atomic_number

        properties = _combine_properties(atom, channels)
        for channel in channels:
            property_grid[channel][idx] = properties[channel]
        property_grid["partial_charge"][idx] = properties["partial_charge"]

        sigma = _atomic_sigma(atom.atomic_number)
        density_grid += _gaussian_density(density_grid, idx, sigma)

    type_grid = _upsample_xy(type_grid, xy_resolution)
    property_grid = {channel: _upsample_xy(grid, xy_resolution) for channel, grid in property_grid.items()}
    density_grid = _upsample_xy(density_grid, xy_resolution)

    density_sum = float(density_grid.sum())
    if density_sum > 0:
        density_grid *= 0.99 / density_sum

    stacked_properties = np.stack(list(property_grid.values()), axis=-1)
    position_matrix = [atom.position for atom in antibiotic.atoms]
    return MoleculeTensors(
        position_matrix=position_matrix,
        type_matrix=type_grid.tolist(),
        property_matrix=stacked_properties.tolist(),
        density_matrix=density_grid.tolist(),
    )


def aggregate_super_matrices(tensors: Iterable[MoleculeTensors]) -> MoleculeTensors:
    tensors_list = list(tensors)
    type_arrays = [np.array(t.type_matrix, dtype=float) for t in tensors_list]
    property_arrays = [np.array(t.property_matrix, dtype=float) for t in tensors_list]
    density_arrays = [np.array(t.density_matrix, dtype=float) for t in tensors_list]

    max_shape = np.max(np.array([arr.shape for arr in type_arrays]), axis=0)
    max_shape_tuple = (int(max_shape[0]), int(max_shape[1]), int(max_shape[2]))

    def _pad_array(array: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
        padded = np.zeros(target_shape + array.shape[3:])
        slices = tuple(slice(0, s) for s in array.shape[:3]) + (slice(None),)
        padded[slices] = array
        return padded

    type_arrays_padded = [_pad_array(arr[..., np.newaxis], max_shape_tuple)[..., 0] for arr in type_arrays]
    property_arrays_padded = [_pad_array(arr, max_shape_tuple) for arr in property_arrays]
    density_arrays_padded = [_pad_array(arr[..., np.newaxis], max_shape_tuple)[..., 0] for arr in density_arrays]

    type_mean = np.mean(np.stack(type_arrays_padded), axis=0)
    property_mean = np.mean(np.stack(property_arrays_padded), axis=0)
    density_mean = np.mean(np.stack(density_arrays_padded), axis=0)

    position_matrix: List[Tuple[float, float, float]] = []
    for tensor in tensors_list:
        position_matrix.extend(tensor.position_matrix)

    return MoleculeTensors(
        position_matrix=position_matrix,
        type_matrix=type_mean.tolist(),
        property_matrix=property_mean.tolist(),
        density_matrix=density_mean.tolist(),
    )

