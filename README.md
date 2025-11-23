# Antibiotic SMD Tensor Representation

This project builds 3D tensor matrices that encode the spatial and electronic structure of antibiotic small-molecule drugs (SMDs). Each molecule yields:

- **Position matrix**: Cartesian coordinates of all atoms relative to a backbone-based origin.
- **Type matrix**: A grid with atomic numbers placed at atom voxels.
- **Property matrix**: Overlaid functional data per voxel (hydrogen bonding, electrophile/nucleophile tendencies, partial charges).
- **Electron density matrix**: Gaussian isosurfaces per atom normalized so the summed probability mass is 0.99.
- **Super matrices**: Mean tensors aggregated per mechanism of action (MoA) and across all molecules.
All grids are upsampled to a 1024×1024 XY resolution (configurable) so downstream visualization and learning consume a consis
tent size; the Z depth is determined by the atomic extent and padding margin.

## How the mathematics is built from first principles

1. **Coordinate frame**: Atom positions \(p_i = (x_i, y_i, z_i)\) are expressed in ångströms relative to the most massive backbone atom (e.g., a beta-lactam carbonyl carbon). This anchors every tensor to a chemically meaningful origin.
2. **Grid construction**: A cubic lattice is built around the atoms with spacing \(\Delta = 0.5\,\text{Å}\) (configurable) and margin \(m = 2.0\,\text{Å}\). Grid bounds are \([\lfloor (p_{\min}-m)/\Delta \rfloor, \lceil (p_{\max}+m)/\Delta \rceil]\).
3. **Type matrix overlay**: At each atom index \(v_i\) the atomic number \(Z_i\) is stored. All other voxels are zero. This explicitly ties nuclear identity to spatial coordinates.
4. **Property channels**: Functional annotations (hydrogen donor/acceptor, electrophile, nucleophile, partial charge) share the same voxel indices as the type grid. Values default to zero and are set per atom so that chemical roles are co-registered with atom positions.
5. **Electron density**: Each atom contributes a normalized Gaussian centered on its voxel index:

   \[
   \rho_i(\mathbf{r}) = \alpha_i \exp\left(-\frac{\lVert \mathbf{r} - v_i \rVert^2}{2\sigma_i^2}\right) \quad \text{with} \quad \sum_{\mathbf{r}} \rho_i(\mathbf{r}) = 0.99
   \]

   The variance \(\sigma_i\) is a simple heuristic derived from covalent radii trends (larger for heavier atoms). Scaling factor \(\alpha_i\) enforces the 0.99 normalization so the discretized cloud integrates to near-unit probability mass.
6. **Aggregation into super matrices**: Molecule tensors are zero-padded to a common lattice, then averaged elementwise. The resulting super matrix captures a typical spatial/electronic pattern for an MoA class and for the full library.
7. **Paper slices**: Each tensor can be visualized as stacked slices (constant-\(x\), constant-\(y\), or constant-\(z\)) laid over one another. Overlaying type, property, and density slices shows how electronic probability density aligns with pharmacophoric features, providing a hand-drawable justification of the model.

## Running the generator

1. Ensure Python 3.11+ is available, then install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Populate `data/antibiotics.json` with antibiotics, their mechanisms, and atom annotations (a starter file is included).
3. Generate tensors (default XY resolution is 1024):

```bash
python scripts/generate_matrices.py --input data/antibiotics.json --output outputs --spacing 0.5 --margin 2.0 --xy-resolution
 1024
```

The script writes JSON tensors for each molecule and super matrices per MoA and globally. Outputs include `*_position.json`, `*_type.json`, `*_property.json`, and `*_density.json`.

Example snippet for `data/antibiotics.json`:

```json
[
  {
    "name": "ampicillin",
    "mechanism": "cell_wall",
    "atoms": [
      {"label": "C1", "atomic_number": 6, "position": [0.0, 0.0, 0.0], "partial_charge": -0.1, "properties": {"h_acceptor": 1}},
      {"label": "N1", "atomic_number": 7, "position": [1.3, 0.0, 0.0], "partial_charge": -0.2, "properties": {"nucleophile": 1}}
    ]
  }
]
```

## File formats and downstream use

- **Molecule tensors**: JSON arrays compatible with common tensor frameworks. Each property matrix stacks channels in the order provided during generation.
- **Super matrices**: Averaged tensors that can seed coarse-grained comparisons or LLM tokenization.
- **Extensibility**: SwissParam and GROMACS descriptors can be merged into the property channels before aggregation, enabling protein-binding simulations or LLM conditioning.

## Conceptual roadmap toward generative design

- Expand property channels with force-field terms (charges, Lennard-Jones radii) from SwissParam outputs.
- Pair tensor slices with text explanations to train an LLM that links spatial/electronic patterns to mechanisms of action.
- Use the super matrices as priors for proposing new atoms or functional group placements, guiding novel antibiotic design.

