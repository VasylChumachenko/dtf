# DFT Python Project - g-C₃N₄ Photocatalyst Analysis

A Python project for Density Functional Theory (DFT) calculations using GPAW, focused on screening g-C₃N₄-based photocatalysts for hydrogen evolution reaction (HER).

## Features

- **Geometry optimization** with GPAW (PW or LCAO mode)
- **DOS calculation** → Band gap, VBM, CBM
- **Vacuum alignment** → Band positions vs vacuum level
- **ΔG_H* calculation** → HER activity descriptor
- **Formation energy** → Defect stability
- **Charge analysis** → Hirshfeld/Bader charges
- **Defect creation** → N-vacancies (bridging/central)
- **Vacancy passivation** → Functional groups (-H, -OH, -NH₂, =O, -CN)
- **Quality presets** → draft/standard/production modes for speed vs accuracy tradeoff

## Setup

1. Create and activate the virtual environment:
```bash
python3 -m venv dftenv
source dftenv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Workflow Overview

```
1. Build structure      → build_g_c3n4.py
2. Create defect        → create_vacancy.py
3. Passivate (optional) → passivate_vacancy.py
4. Optimize geometry    → optimize_geometry.py
5. Calculate properties → calculate_properties.py
6. H adsorption         → calculate_h_adsorption.py
7. Charge analysis      → calculate_charges.py
   
Or use: run_full_analysis.py (all-in-one after optimization)
```

## Quality Presets

All analysis scripts support `--quality` flag for speed vs accuracy tradeoff:

| Preset | ecut | k-points | Convergence | Speedup | Use Case |
|--------|------|----------|-------------|---------|----------|
| `draft` | 300 eV | 2×2×1 / 3×3×1 | 1e-3 | ~5× | Initial screening |
| `standard` | 500 eV | 4×4×1 / 6×6×1 | 1e-4 | 1× | Normal workflow |
| `production` | 600 eV | 8×8×1 / 12×12×1 | 1e-6 | ~0.3× | Publication |

```bash
# Fast screening of many structures
gpaw python run_full_analysis.py structure.cif --quality draft

# Standard calculation (default)
gpaw python run_full_analysis.py structure.cif

# Publication quality
gpaw python run_full_analysis.py structure.cif --quality production
```

Draft mode results are clearly labeled with uncertainty estimates (~±0.15 eV).

## Scripts

### 1. Build g-C₃N₄ Structure
```bash
python build_g_c3n4.py
```

### 2. Create N-Vacancy
```bash
# Show available N atoms
python create_vacancy.py g_c3n4_monolayer.cif

# Create bridging N vacancy in 2×2 supercell
python create_vacancy.py g_c3n4_monolayer.cif --supercell 2 2 1 --type bridging

# Create central N vacancy
python create_vacancy.py g_c3n4_monolayer.cif --supercell 2 2 1 --type central
```

### 2b. Passivate Vacancy (Optional)
```bash
# Auto-detect undercoordinated atoms and passivate with H
python passivate_vacancy.py vacancy.cif --auto --group H

# Passivate with other functional groups
python passivate_vacancy.py vacancy.cif --auto --group OH   # hydroxyl
python passivate_vacancy.py vacancy.cif --auto --group NH2  # amino
python passivate_vacancy.py vacancy.cif --auto --group O    # oxo (=O)
python passivate_vacancy.py vacancy.cif --auto --group CN   # cyano

# Interactive mode (choose site-by-site)
python passivate_vacancy.py vacancy.cif
```

Available functional groups: `-H`, `-OH`, `-NH₂`, `=O`, `-CN`

### 3. Geometry Optimization
```bash
# Serial (PW mode - default, accurate)
gpaw python optimize_geometry.py structure.cif --fix-c --kpts 4 4 1

# Parallel with fast pre-optimization
mpirun -n 4 gpaw python -- optimize_geometry.py structure.cif --fix-c --kpts 4 4 1 --fast

# LCAO mode (much faster, good for initial screening)
mpirun -n 4 gpaw python -- optimize_geometry.py structure.cif --lcao --fix-cell

# FIRE-only optimization (robust for problematic structures)
mpirun -n 4 gpaw python -- optimize_geometry.py structure.cif --lcao --fire-only --fix-cell
```

Options:
- `--fix-c` : Fix c-axis only (for 2D materials with vacuum)
- `--fix-cell` : Fix all cell parameters (recommended for defects)
- `--fast` : Coarse pre-optimization then refine
- `--lcao` : Use LCAO mode (faster, ~5-10× speedup)
- `--basis dzp` : LCAO basis set (szp, dzp, tzp)
- `--fire-only` : Use FIRE optimizer only (more robust for defects)
- `--kpts 4 4 1` : k-point grid
- `--ecut 500` : Plane-wave cutoff (eV, PW mode only)
- `--fmax 0.03` : Force convergence (eV/Å)

### 4. DOS and Electronic Properties
```bash
# Fast draft calculation (~5× faster)
gpaw python calculate_properties.py optimized.cif --quality draft

# Standard DOS calculation
gpaw python calculate_properties.py optimized.cif

# With formation energy (defect vs pristine)
gpaw python calculate_properties.py defect.cif --pristine pristine.cif

# Parallel (production quality)
mpirun -n 4 gpaw python -- calculate_properties.py optimized.cif --quality production
```

**Output:**
- `*_dos.png` : DOS plot
- `*_dos.json` : Band gap, VBM, CBM, vacuum alignment
- `*_properties.json` : All computed properties

### 5. H Adsorption (ΔG_H*)
```bash
# Find adsorption sites
python calculate_h_adsorption.py structure.cif --find-sites

# Fast screening of multiple sites
gpaw python calculate_h_adsorption.py structure.cif --site-index 15 --quality draft

# Standard calculation
gpaw python calculate_h_adsorption.py structure.cif --add-h 5.0 3.0 12.5

# Production quality with pre-computed surface energy (saves time)
gpaw python calculate_h_adsorption.py structure.cif --site-index 15 \
    --surface-energy -450.123 --quality production

# Calculate H2 reference energy (do once)
gpaw python calculate_h_adsorption.py --calc-h2
```

**Interpretation:**
- |ΔG_H*| < 0.1 eV → Excellent HER activity
- |ΔG_H*| < 0.2 eV → Good activity
- |ΔG_H*| > 0.5 eV → Poor activity

### 6. Charge Analysis
```bash
# Hirshfeld charges (built-in GPAW)
gpaw python calculate_charges.py structure.cif --method hirshfeld

# Export for external Bader analysis
gpaw python calculate_charges.py structure.cif --export-cube
# Then: bader density.cube
# Parse: python calculate_charges.py structure.cif --parse-bader ACF.dat
```

### 7. Full Analysis (All-in-One)
```bash
# Fast screening (~5× speedup)
gpaw python run_full_analysis.py structure.cif --quality draft --dos-only

# Complete workflow (standard)
gpaw python run_full_analysis.py defect.cif --pristine pristine.cif --h-site 15

# Publication quality
mpirun -n 4 gpaw python -- run_full_analysis.py defect.cif \
    --pristine pristine.cif --h-site 15 --quality production
```

**Output:**
- `*_summary.json` : All properties
- `*_report.txt` : Human-readable report
- `*_dos.png` : DOS plot

## Key Properties for Literature Comparison

| Property | Symbol | Units | Target |
|----------|--------|-------|--------|
| Band gap | E_g | eV | 2.5-2.8 (pristine g-C₃N₄) |
| CBM vs vacuum | E_CBM | eV | > -4.44 (HER possible) |
| VBM vs vacuum | E_VBM | eV | < -5.67 (OER possible) |
| H adsorption free energy | ΔG_H* | eV | ≈ 0 (optimal HER) |
| Formation energy | E_f | eV | Stability indicator |
| Bader charge | q | e | Charge on defect site |

## Example: Complete Defect Analysis

```bash
# 1. Create defect structure
python create_vacancy.py g_c3n4_monolayer.cif --supercell 2 2 1 --type bridging

# 2. Passivate dangling bonds (optional but recommended)
python passivate_vacancy.py g_c3n4_monolayer_2x2x1_Nvac_bridging.cif --auto --group H

# 3. Optimize defect geometry (LCAO for speed, fix cell for stability)
mpirun -n 8 gpaw python -- optimize_geometry.py \
    g_c3n4_monolayer_2x2x1_Nvac_bridging_passivated_H.cif \
    --lcao --fix-cell --quality draft

# 4. Also optimize pristine supercell (for formation energy)
mpirun -n 8 gpaw python -- optimize_geometry.py \
    g_c3n4_monolayer.cif --supercell 2 2 1 \
    --lcao --fix-cell --quality draft

# 5. Run full analysis
mpirun -n 8 gpaw python -- run_full_analysis.py \
    g_c3n4_monolayer_2x2x1_Nvac_bridging_passivated_H/..._draft_optimized.cif \
    --pristine g_c3n4_monolayer_2x2x1/..._draft_optimized.cif \
    --h-site 15 \
    --quality draft
```

### Fast Screening Workflow (Draft Quality)

For initial MVD (Minimal Viable Dataset) screening:

```bash
# Quick geometry optimization with LCAO
mpirun -n 8 gpaw python -- optimize_geometry.py structure.cif \
    --lcao --fix-cell --quality draft --fire-only

# Quick property analysis
mpirun -n 8 gpaw python -- run_full_analysis.py optimized.cif --quality draft
```

Draft mode provides ~5× speedup with clearly labeled uncertainty estimates.

## Output Files Summary

| File | Contents |
|------|----------|
| `*_optimized.cif` | Optimized structure |
| `*_dos.png` | DOS plot with band edges |
| `*_dos.json` | DOS data (energies, band gap, VBM, CBM) |
| `*_properties.json` | All electronic properties |
| `*_h_adsorption.json` | ΔG_H* results |
| `*_hirshfeld_charges.json` | Atomic charges |
| `*_summary.json` | Complete property summary |
| `*_report.txt` | Human-readable report |

## References

- ΔG_H* methodology: Nørskov et al., J. Electrochem. Soc. 152, J23 (2005)
- Bader analysis: Henkelman et al., Comput. Mater. Sci. 36, 354 (2006)
- g-C₃N₄ photocatalysis: Wang et al., Nat. Mater. 8, 76 (2009)

## Requirements

- Python 3.x
- GPAW (Grid-based Projector-Augmented Wave)
- ASE (Atomic Simulation Environment)
- NumPy, SciPy, Matplotlib
- mpi4py (for parallel calculations)

## License

[Add your license here]
