#!/usr/bin/env python3
"""
Create N-vacancy in g-C₃N₄ structure.

Usage:
    python create_vacancy.py structure.cif                    # Interactive: show N atoms
    python create_vacancy.py structure.cif -i 5               # Remove N at index 5
    python create_vacancy.py structure.cif --type bridging    # Remove a bridging N
    python create_vacancy.py structure.cif -i 5 -o defect.cif # Save to file

N-vacancy types in g-C₃N₄:
    - N₃c (central): Inside heptazine ring, 3-coordinated to C
    - N₂c (bridging): Connects heptazine units, 2-coordinated to C
"""

import argparse
import numpy as np
from ase import io, Atoms
from ase.neighborlist import neighbor_list


def get_coordination(atoms, cutoff=1.6):
    """Get coordination number for each atom."""
    i_list, j_list = neighbor_list('ij', atoms, cutoff=cutoff)
    
    coordination = np.zeros(len(atoms), dtype=int)
    for i in i_list:
        coordination[i] += 1
    
    return coordination


def classify_nitrogen(atoms, cutoff=1.6):
    """
    Classify N atoms in g-C₃N₄ structure.
    
    Returns dict with:
        - 'central': indices of N₃c (3-coordinated, central)
        - 'bridging': indices of N₂c (2-coordinated, bridging)
        - 'other': any other N atoms
    """
    symbols = atoms.get_chemical_symbols()
    coordination = get_coordination(atoms, cutoff)
    
    # Get neighbor types for each atom
    i_list, j_list = neighbor_list('ij', atoms, cutoff=cutoff)
    
    nitrogen_types = {
        'central': [],   # N₃c - coordinated to 3 C
        'bridging': [],  # N₂c - coordinated to 2 C
        'other': []
    }
    
    for idx, (sym, coord) in enumerate(zip(symbols, coordination)):
        if sym != 'N':
            continue
            
        # Count C neighbors
        c_neighbors = sum(1 for j in j_list[i_list == idx] if symbols[j] == 'C')
        
        if coord == 3 and c_neighbors == 3:
            nitrogen_types['central'].append(idx)
        elif coord == 2 and c_neighbors == 2:
            nitrogen_types['bridging'].append(idx)
        else:
            nitrogen_types['other'].append(idx)
    
    return nitrogen_types


def print_nitrogen_info(atoms, n_types):
    """Print information about N atoms."""
    symbols = atoms.get_chemical_symbols()
    positions = atoms.positions
    
    print("\n" + "="*60)
    print("NITROGEN ATOMS IN STRUCTURE")
    print("="*60)
    
    print(f"\nTotal atoms: {len(atoms)}")
    print(f"Total N atoms: {symbols.count('N')}")
    print(f"  - Central (N₃c): {len(n_types['central'])}")
    print(f"  - Bridging (N₂c): {len(n_types['bridging'])}")
    if n_types['other']:
        print(f"  - Other: {len(n_types['other'])}")
    
    print("\n--- Central N atoms (N₃c, inside heptazine ring) ---")
    for idx in n_types['central']:
        pos = positions[idx]
        print(f"  Index {idx:3d}: ({pos[0]:7.3f}, {pos[1]:7.3f}, {pos[2]:7.3f})")
    
    print("\n--- Bridging N atoms (N₂c, connects heptazines) ---")
    for idx in n_types['bridging']:
        pos = positions[idx]
        print(f"  Index {idx:3d}: ({pos[0]:7.3f}, {pos[1]:7.3f}, {pos[2]:7.3f})")
    
    if n_types['other']:
        print("\n--- Other N atoms ---")
        for idx in n_types['other']:
            pos = positions[idx]
            print(f"  Index {idx:3d}: ({pos[0]:7.3f}, {pos[1]:7.3f}, {pos[2]:7.3f})")
    
    print("="*60)


def create_vacancy(atoms, vacancy_index):
    """
    Create vacancy by removing atom at given index.
    
    Returns new Atoms object without the specified atom.
    """
    # Get all indices except the vacancy
    mask = np.ones(len(atoms), dtype=bool)
    mask[vacancy_index] = False
    
    # Create new atoms object
    new_atoms = atoms[mask]
    
    return new_atoms


def main():
    parser = argparse.ArgumentParser(
        description="Create N-vacancy in g-C₃N₄ structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("cif_file", help="Input CIF file")
    parser.add_argument("-i", "--index", type=int, default=None,
                        help="Index of N atom to remove")
    parser.add_argument("--type", choices=["central", "bridging"],
                        default=None,
                        help="Type of N to remove (picks first of that type)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output CIF file (default: input_Nvac.cif)")
    parser.add_argument("--supercell", type=int, nargs=3, default=None,
                        metavar=("NX", "NY", "NZ"),
                        help="Create supercell before vacancy (recommended: 2 2 1)")
    
    args = parser.parse_args()
    
    # Read structure
    print(f"Reading: {args.cif_file}")
    atoms = io.read(args.cif_file, format='cif')
    
    # Create supercell if requested
    if args.supercell:
        sc = tuple(args.supercell)
        n_orig = len(atoms)
        atoms = atoms * sc
        print(f"Created {sc[0]}×{sc[1]}×{sc[2]} supercell: {n_orig} → {len(atoms)} atoms")
    
    # Classify N atoms
    n_types = classify_nitrogen(atoms)
    
    # Print info
    print_nitrogen_info(atoms, n_types)
    
    # Determine which atom to remove
    vacancy_index = None
    vacancy_type = None
    
    if args.index is not None:
        vacancy_index = args.index
        symbols = atoms.get_chemical_symbols()
        if symbols[vacancy_index] != 'N':
            print(f"\nERROR: Atom at index {vacancy_index} is {symbols[vacancy_index]}, not N!")
            return
        # Determine type
        if vacancy_index in n_types['central']:
            vacancy_type = 'central'
        elif vacancy_index in n_types['bridging']:
            vacancy_type = 'bridging'
        else:
            vacancy_type = 'other'
            
    elif args.type is not None:
        if n_types[args.type]:
            vacancy_index = n_types[args.type][0]  # Take first of that type
            vacancy_type = args.type
        else:
            print(f"\nERROR: No {args.type} N atoms found in structure!")
            return
    else:
        # Interactive mode - just show info
        print("\nTo create vacancy, run again with:")
        print(f"  python create_vacancy.py {args.cif_file} -i <index>")
        print(f"  python create_vacancy.py {args.cif_file} --type bridging")
        print(f"\nRecommended: Use 2×2×1 supercell for dilute vacancy:")
        print(f"  python create_vacancy.py {args.cif_file} --supercell 2 2 1 --type bridging")
        return
    
    # Create vacancy
    print(f"\n>>> Creating vacancy at index {vacancy_index} ({vacancy_type} N)")
    
    removed_pos = atoms.positions[vacancy_index]
    print(f"    Position: ({removed_pos[0]:.3f}, {removed_pos[1]:.3f}, {removed_pos[2]:.3f})")
    
    defect_atoms = create_vacancy(atoms, vacancy_index)
    
    print(f"\nDefect structure:")
    print(f"  Formula: {defect_atoms.get_chemical_formula()}")
    print(f"  Atoms: {len(atoms)} → {len(defect_atoms)}")
    
    # Check geometry
    from ase.neighborlist import neighbor_list
    i_list, j_list, d_list = neighbor_list('ijd', defect_atoms, cutoff=3.0)
    if len(d_list) > 0:
        print(f"  Min distance: {d_list.min():.3f} Å")
    
    # Generate output filename
    if args.output:
        output_file = args.output
    else:
        base = args.cif_file.replace('.cif', '').replace('.txt', '')
        if args.supercell:
            sc = args.supercell
            base = f"{base}_{sc[0]}x{sc[1]}x{sc[2]}"
        output_file = f"{base}_Nvac_{vacancy_type}.cif"
    
    # Save
    io.write(output_file, defect_atoms)
    print(f"\nSaved: {output_file}")
    
    print("\n>>> Next steps:")
    print(f"  1. Optimize the defect structure:")
    print(f"     mpirun -n 8 gpaw python -- optimize_geometry.py {output_file} --fix-c --kpts 3 3 1")
    print(f"  2. Compare energy with pristine structure to get vacancy formation energy")


if __name__ == "__main__":
    main()




