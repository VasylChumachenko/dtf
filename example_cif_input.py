"""
Example script showing how to use CIF files as input for GPAW calculations.
"""

from ase import io
from gpaw import GPAW, PW
from gpaw import FermiDirac

# Read structure from CIF file
atoms = io.read('structure.cif')

# Optional: Print information about the structure
print(f"Number of atoms: {len(atoms)}")
print(f"Chemical formula: {atoms.get_chemical_formula()}")
print(f"Cell parameters: {atoms.cell.lengths()}")

# Set up GPAW calculator
calc = GPAW(
    mode=PW(400),              # Plane wave basis with 400 eV cutoff
    xc='PBE',                  # Exchange-correlation functional
    occupations=FermiDirac(0.1),  # Smearing
    kpts=(4, 4, 4),            # k-point grid
    txt='output.txt'           # Output file
)

# Attach calculator to atoms
atoms.calc = calc

# Run calculation (e.g., get total energy)
energy = atoms.get_potential_energy()
print(f"Total energy: {energy:.4f} eV")

# You can also write the structure to other formats
# io.write('output.xyz', atoms)







