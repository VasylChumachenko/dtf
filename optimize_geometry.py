"""
Geometry optimization script for CIF file using GPAW.

This script reads a CIF file, performs geometry optimization
(both atomic positions and cell parameters), and saves the optimized structure.

Usage:
    gpaw python optimize_geometry.py <input.cif>
    mpirun -n 4 gpaw python optimize_geometry.py <input.cif>

Example:
    gpaw python optimize_geometry.py C3N4_mp-1193580_primitive.cif.txt
"""

import argparse
import os
from ase import io
from ase.optimize import LBFGS
from ase.constraints import UnitCellFilter
from ase.parallel import parprint, paropen, world
from gpaw import GPAW, PW
from gpaw import FermiDirac


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Geometry optimization using GPAW'
    )
    parser.add_argument(
        'cif_file',
        nargs='?',
        default='C3N4_mp-1193580_primitive.cif.txt',
        help='Input CIF file (default: C3N4_mp-1193580_primitive.cif.txt)'
    )
    args = parser.parse_args()

    cif_file = args.cif_file

    # Generate output filenames based on input
    base_name = os.path.splitext(os.path.basename(cif_file))[0]
    # Remove .cif if it's a .cif.txt file
    if base_name.endswith('.cif'):
        base_name = base_name[:-4]
    output_cif = f'{base_name}_optimized.cif'
    output_xyz = f'{base_name}_optimized.xyz'
    gpaw_log = f'{base_name}_gpaw.log'
    traj_file = f'{base_name}_optimization.traj'

    parprint(f"Reading structure from {cif_file}...")
    atoms = io.read(cif_file, format='cif')

    # Print initial structure information
    parprint(f"\nInitial structure:")
    parprint(f"  Number of atoms: {len(atoms)}")
    parprint(f"  Chemical formula: {atoms.get_chemical_formula()}")
    parprint(f"  Cell parameters (a, b, c): {atoms.cell.lengths()}")
    parprint(f"  Cell angles (α, β, γ): {atoms.cell.angles()}")
    parprint(f"  Cell volume: {atoms.get_volume():.2f} Å³")

    # Set up GPAW calculator
    parprint("\nSetting up GPAW calculator...")

    # For MPI: explicitly use k-point parallelization to avoid domain decomposition memory overhead
    parallel_opts = None
    if world.size > 1:
        # Force k-point parallelization (NOT domain decomposition)
        # Domain decomposition can cause memory bloat for small systems
        parallel_opts = {'kpt': world.size, 'domain': 1, 'band': 1}
        parprint(f"  Using k-point parallelization with {world.size} groups")

    calc = GPAW(
        mode=PW(400),              # Plane wave basis with 400 eV cutoff
        xc='PBE',                   # PBE exchange-correlation functional
        occupations=FermiDirac(0.1),  # Fermi-Dirac smearing with 0.1 eV width
        kpts=(4, 4, 4),             # k-point grid
        txt=gpaw_log,               # GPAW output
        convergence={'energy': 1e-5, 'density': 1e-6},
        symmetry='off',             # Required for cell optimization
        parallel=parallel_opts      # K-point parallelization (avoids domain memory overhead)
    )

    atoms.calc = calc

    # Get initial energy
    parprint("\nCalculating initial energy...")
    initial_energy = atoms.get_potential_energy()
    parprint(f"  Initial energy: {initial_energy:.6f} eV")

    # Perform geometry optimization
    parprint("\nStarting geometry optimization...")
    parprint("  Optimizing atomic positions and cell parameters...")

    # Use UnitCellFilter to optimize both atomic positions and cell
    ucf = UnitCellFilter(atoms)

    # Create optimizer (use '-' for stdout to avoid file conflicts in MPI)
    optimizer = LBFGS(ucf, trajectory=traj_file, logfile='-')

    # Run optimization (fmax = force convergence criterion in eV/Å)
    optimizer.run(fmax=0.05)

    # Get final energy
    final_energy = atoms.get_potential_energy()
    parprint(f"\nOptimization complete!")
    parprint(f"  Final energy: {final_energy:.6f} eV")
    parprint(f"  Energy change: {final_energy - initial_energy:.6f} eV")

    # Print final structure information
    parprint(f"\nFinal structure:")
    parprint(f"  Cell parameters (a, b, c): {atoms.cell.lengths()}")
    parprint(f"  Cell angles (α, β, γ): {atoms.cell.angles()}")
    parprint(f"  Cell volume: {atoms.get_volume():.2f} Å³")

    # Calculate forces to verify convergence
    forces = atoms.get_forces()
    max_force = (forces**2).sum(axis=1).max()**0.5
    parprint(f"  Maximum force: {max_force:.6f} eV/Å")

    # Save optimized structure
    io.write(output_cif, atoms)
    parprint(f"\nOptimized structure saved to: {output_cif}")

    io.write(output_xyz, atoms)
    parprint(f"Also saved as: {output_xyz}")

    parprint("\nDone!")


if __name__ == '__main__':
    main()
