# DFT Python Project

A Python project for Density Functional Theory (DFT) calculations using GPAW.

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

## Requirements

- Python 3.x
- GPAW (GPAW - Grid-based Projector-Augmented Wave)
- ASE (Atomic Simulation Environment)
- NumPy
- SciPy
- Matplotlib

## Usage

Activate the virtual environment before running any scripts:
```bash
source dftenv/bin/activate
```

### Running Geometry Optimization

The script `optimize_geometry.py` follows GPAW documentation best practices for parallel runs.

**Serial mode (no MPI):**
```bash
python optimize_geometry.py
```

**With MPI (recommended for larger systems):**
```bash
# Using gpaw command (recommended)
gpaw -P 4 python optimize_geometry.py

# Or using mpiexec
mpiexec -n 4 gpaw python optimize_geometry.py
```

**Hybrid OpenMP/MPI (for better performance):**
```bash
export OMP_NUM_THREADS=2
mpiexec -n 8 gpaw python optimize_geometry.py
# Total: 8 processes × 2 threads = 16 cores
```

**Using the helper script:**
```bash
./run_optimization.sh              # Serial
./run_optimization.sh -n 4         # 4 MPI processes
./run_optimization.sh -n 8 -t 2    # Hybrid: 8 MPI × 2 threads
```

**Check parallelization setup (dry-run):**
```bash
gpaw python --dry-run=8 optimize_geometry.py
```
This shows how GPAW would parallelize with 8 processes and estimates memory usage.

For more details, see the [GPAW parallel runs documentation](https://gpaw.readthedocs.io/documentation/parallel_runs/parallel_runs.html).

## CIF File Support

Yes! GPAW can use CIF (Crystallographic Information File) files as input through ASE. 

Example usage:
```python
from ase import io
from gpaw import GPAW, PW

# Read structure from CIF file
atoms = io.read('structure.cif')

# Set up GPAW calculator
calc = GPAW(mode=PW(400), xc='PBE', kpts=(4, 4, 4))
atoms.calc = calc

# Run calculation
energy = atoms.get_potential_energy()
```

See `example_cif_input.py` for a complete example.

## License

[Add your license here]
