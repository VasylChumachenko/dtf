#!/usr/bin/env python3
"""
Passivate dangling bonds in vacancy structures with functional groups.

This script detects undercoordinated atoms (typically C atoms after N-vacancy)
and allows interactive assignment of functional groups to passivate them.

Supported groups:
    -H    : Hydrogen (simplest passivation)
    -OH   : Hydroxyl (aqueous/humid conditions)
    -NH2  : Amine (NH3 synthesis atmosphere)
    =O    : Oxygen/ketone (oxidizing conditions)
    -CN   : Cyano (carbon nitride decomposition)

Usage:
    python passivate_vacancy.py structure_with_vacancy.cif
    python passivate_vacancy.py structure.cif --detect-only
    python passivate_vacancy.py structure.cif --auto H  # passivate all with H

After passivation, run geometry optimization:
    mpirun -n 4 gpaw python -- optimize_geometry.py passivated_structure.cif
"""

import argparse
import sys
import numpy as np
from ase import Atoms, Atom
from ase import io
from ase.neighborlist import NeighborList, natural_cutoffs


# =============================================================================
# Functional Group Templates
# =============================================================================

# Bond lengths in Angstroms (typical values)
BOND_LENGTHS = {
    'C-H': 1.09,
    'C-O': 1.43,  # single bond
    'C=O': 1.23,  # double bond (ketone)
    'C-N': 1.47,  # single bond
    'C≡N': 1.16,  # triple bond (cyano)
    'O-H': 0.96,
    'N-H': 1.01,
}

# Expected coordination numbers
EXPECTED_COORDINATION = {
    'C': 3,  # sp2 carbon in g-C3N4
    'N': 2,  # bridging N (N2c) or 3 for central N (N3c)
}


def get_group_atoms(group_type, bond_vector):
    """
    Generate atomic positions for a functional group.
    
    Args:
        group_type: One of 'H', 'OH', 'NH2', 'O', 'CN'
        bond_vector: Unit vector pointing from C toward where group should attach
        
    Returns:
        List of (symbol, position_relative_to_C) tuples
    """
    # Normalize bond vector
    bond_vector = np.array(bond_vector)
    bond_vector = bond_vector / np.linalg.norm(bond_vector)
    
    # Create perpendicular vectors for multi-atom groups
    if abs(bond_vector[2]) < 0.9:
        perp1 = np.cross(bond_vector, [0, 0, 1])
    else:
        perp1 = np.cross(bond_vector, [1, 0, 0])
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(bond_vector, perp1)
    
    atoms_to_add = []
    
    if group_type == 'H':
        # Simple hydrogen
        pos = bond_vector * BOND_LENGTHS['C-H']
        atoms_to_add.append(('H', pos))
        
    elif group_type == 'OH':
        # Hydroxyl: C-O-H with ~109° angle
        O_pos = bond_vector * BOND_LENGTHS['C-O']
        # H is at angle from O
        H_direction = bond_vector * 0.5 + perp1 * 0.866  # ~109° from C-O bond
        H_direction = H_direction / np.linalg.norm(H_direction)
        H_pos = O_pos + H_direction * BOND_LENGTHS['O-H']
        atoms_to_add.append(('O', O_pos))
        atoms_to_add.append(('H', H_pos))
        
    elif group_type == 'NH2':
        # Amine: C-N with two H atoms, pyramidal geometry
        N_pos = bond_vector * BOND_LENGTHS['C-N']
        # Two H atoms at ~109° from C-N bond
        angle = np.radians(109.5 / 2)  # Half angle for pyramidal
        H1_direction = bond_vector * np.cos(angle) + perp1 * np.sin(angle)
        H2_direction = bond_vector * np.cos(angle) - perp1 * np.sin(angle)
        # Tilt slightly for pyramidal shape
        tilt = perp2 * 0.3
        H1_pos = N_pos + (H1_direction / np.linalg.norm(H1_direction)) * BOND_LENGTHS['N-H'] + tilt
        H2_pos = N_pos + (H2_direction / np.linalg.norm(H2_direction)) * BOND_LENGTHS['N-H'] + tilt
        atoms_to_add.append(('N', N_pos))
        atoms_to_add.append(('H', H1_pos))
        atoms_to_add.append(('H', H2_pos))
        
    elif group_type == 'O':
        # Ketone oxygen (double bond character)
        pos = bond_vector * BOND_LENGTHS['C=O']
        atoms_to_add.append(('O', pos))
        
    elif group_type == 'CN':
        # Cyano: C≡N linear
        C_pos = bond_vector * 1.40  # C-C bond to cyano carbon
        N_pos = C_pos + bond_vector * BOND_LENGTHS['C≡N']
        atoms_to_add.append(('C', C_pos))
        atoms_to_add.append(('N', N_pos))
        
    else:
        raise ValueError(f"Unknown group type: {group_type}")
    
    return atoms_to_add


# =============================================================================
# Detection Functions
# =============================================================================

def find_undercoordinated_atoms(atoms, target_element='C', bond_cutoff=1.6):
    """
    Find atoms with fewer neighbors than expected.
    
    Args:
        atoms: ASE Atoms object
        target_element: Element to check (default 'C')
        bond_cutoff: Maximum bond length to consider (default 1.6 Å for C-N)
        
    Returns:
        List of (index, current_coordination, neighbor_indices) tuples
    """
    # Build neighbor list with fixed cutoff for all atoms
    # For g-C3N4: C-N ~ 1.33 Å, so 1.6 Å cutoff is reasonable
    cutoffs = [bond_cutoff / 2] * len(atoms)  # NeighborList uses sum of cutoffs
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)
    
    undercoordinated = []
    symbols = atoms.get_chemical_symbols()
    
    for i, sym in enumerate(symbols):
        if sym != target_element:
            continue
            
        indices, offsets = nl.get_neighbors(i)
        n_neighbors = len(indices)
        expected = EXPECTED_COORDINATION.get(sym, 4)
        
        if n_neighbors < expected:
            undercoordinated.append({
                'index': i,
                'symbol': sym,
                'coordination': n_neighbors,
                'expected': expected,
                'missing': expected - n_neighbors,
                'neighbors': indices.tolist(),
                'position': atoms.positions[i].copy()
            })
    
    return undercoordinated


def calculate_passivation_vector(atoms, site_info):
    """
    Calculate the direction for placing a functional group.
    
    The vector points away from existing neighbors (into the "vacancy").
    
    Args:
        atoms: ASE Atoms object
        site_info: Dict from find_undercoordinated_atoms
        
    Returns:
        Unit vector pointing toward passivation direction
    """
    center = site_info['position']
    neighbor_indices = site_info['neighbors']
    
    if len(neighbor_indices) == 0:
        # No neighbors - point along z
        return np.array([0, 0, 1])
    
    # Calculate vectors to all neighbors
    neighbor_vectors = []
    for ni in neighbor_indices:
        vec = atoms.positions[ni] - center
        # Handle PBC
        vec = vec - np.round(vec / atoms.cell.lengths()) * atoms.cell.lengths()
        neighbor_vectors.append(vec / np.linalg.norm(vec))
    
    # Passivation vector is opposite to average neighbor direction
    avg_neighbor = np.mean(neighbor_vectors, axis=0)
    
    if np.linalg.norm(avg_neighbor) < 0.1:
        # Neighbors are symmetric - use cross product to find perpendicular
        if len(neighbor_vectors) >= 2:
            passivation = np.cross(neighbor_vectors[0], neighbor_vectors[1])
        else:
            passivation = np.array([0, 0, 1])
    else:
        passivation = -avg_neighbor
    
    # Normalize
    passivation = passivation / np.linalg.norm(passivation)
    
    return passivation


# =============================================================================
# Interactive Menu
# =============================================================================

def print_site_info(sites, atoms):
    """Print information about undercoordinated sites."""
    print("\n" + "=" * 60)
    print("DETECTED UNDERCOORDINATED SITES")
    print("=" * 60)
    
    if not sites:
        print("No undercoordinated atoms found!")
        return
    
    print(f"\nFound {len(sites)} site(s) requiring passivation:\n")
    
    for i, site in enumerate(sites):
        neighbors = [atoms.get_chemical_symbols()[n] for n in site['neighbors']]
        print(f"  Site {i+1}:")
        print(f"    Atom index:    {site['index']}")
        print(f"    Element:       {site['symbol']}")
        print(f"    Position:      ({site['position'][0]:.3f}, {site['position'][1]:.3f}, {site['position'][2]:.3f})")
        print(f"    Coordination:  {site['coordination']} (expected {site['expected']})")
        print(f"    Neighbors:     {', '.join(neighbors)} (indices: {site['neighbors']})")
        print(f"    Missing bonds: {site['missing']}")
        print()


def interactive_menu(sites, atoms):
    """
    Interactive menu for assigning functional groups to sites.
    
    Returns:
        Dict mapping site index to group type
    """
    assignments = {}
    
    print("\n" + "=" * 60)
    print("FUNCTIONAL GROUP ASSIGNMENT")
    print("=" * 60)
    print("\nAvailable groups:")
    print("  H    - Hydrogen (-H)")
    print("  OH   - Hydroxyl (-OH)")
    print("  NH2  - Amine (-NH2)")
    print("  O    - Ketone (=O)")
    print("  CN   - Cyano (-CN)")
    print("  skip - Leave this site unpassivated")
    print("  all  - Apply same group to all remaining sites")
    print()
    
    valid_groups = ['H', 'OH', 'NH2', 'O', 'CN', 'skip']
    apply_all = None
    
    for i, site in enumerate(sites):
        if apply_all is not None:
            if apply_all != 'skip':
                assignments[i] = apply_all
            continue
            
        neighbors = [atoms.get_chemical_symbols()[n] for n in site['neighbors']]
        print(f"\nSite {i+1}/{len(sites)}: Atom {site['index']} ({site['symbol']})")
        print(f"  Neighbors: {', '.join(neighbors)}")
        
        while True:
            choice = input(f"  Assign group [H/OH/NH2/O/CN/skip/all]: ").strip().upper()
            
            if choice == 'ALL':
                all_choice = input(f"  Apply which group to all remaining? [H/OH/NH2/O/CN/skip]: ").strip().upper()
                if all_choice in valid_groups:
                    apply_all = all_choice
                    if apply_all != 'SKIP':
                        assignments[i] = apply_all
                    break
                else:
                    print(f"  Invalid choice. Use: {', '.join(valid_groups)}")
            elif choice in valid_groups:
                if choice != 'SKIP':
                    assignments[i] = choice
                break
            else:
                print(f"  Invalid choice. Use: {', '.join(valid_groups + ['all'])}")
    
    return assignments


# =============================================================================
# Passivation
# =============================================================================

def add_functional_groups(atoms, sites, assignments):
    """
    Add functional groups to the structure.
    
    Args:
        atoms: ASE Atoms object (modified in place)
        sites: List of site info dicts
        assignments: Dict mapping site index to group type
        
    Returns:
        Modified atoms object, list of added atom indices
    """
    added_atoms = []
    atoms_to_append = []
    
    for site_idx, group_type in assignments.items():
        site = sites[site_idx]
        
        # Calculate passivation direction
        pass_vector = calculate_passivation_vector(atoms, site)
        
        # Get atoms for this group
        group_atoms = get_group_atoms(group_type, pass_vector)
        
        # Add atoms relative to the undercoordinated site
        base_pos = site['position']
        for symbol, rel_pos in group_atoms:
            abs_pos = base_pos + rel_pos
            atoms_to_append.append(Atom(symbol, abs_pos))
            added_atoms.append({
                'symbol': symbol,
                'position': abs_pos,
                'group': group_type,
                'attached_to': site['index']
            })
    
    # Append all new atoms
    for atom in atoms_to_append:
        atoms.append(atom)
    
    return atoms, added_atoms


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Passivate dangling bonds with functional groups",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("structure", help="Input structure file (CIF/XYZ)")
    parser.add_argument("--detect-only", action="store_true",
                        help="Only detect undercoordinated sites, don't passivate")
    parser.add_argument("--auto", type=str, metavar="GROUP",
                        help="Automatically passivate all sites with specified group (H/OH/NH2/O/CN)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output file name (default: {input}_passivated.cif)")
    parser.add_argument("--element", type=str, default="C",
                        help="Element to check for undercoordination (default: C)")
    parser.add_argument("--bond-cutoff", type=float, default=1.6,
                        help="Maximum bond length in Angstroms (default: 1.6)")
    
    args = parser.parse_args()
    
    # Read structure
    print(f"\nReading: {args.structure}")
    atoms = io.read(args.structure)
    print(f"Formula: {atoms.get_chemical_formula()}")
    print(f"Atoms:   {len(atoms)}")
    
    # Find undercoordinated sites
    sites = find_undercoordinated_atoms(
        atoms, 
        target_element=args.element,
        bond_cutoff=args.bond_cutoff
    )
    
    # Print site info
    print_site_info(sites, atoms)
    
    if args.detect_only:
        print("\n--detect-only specified, exiting without passivation.")
        return
    
    if not sites:
        print("\nNo sites to passivate. Structure appears fully coordinated.")
        return
    
    # Get assignments
    if args.auto:
        group = args.auto.upper()
        valid = ['H', 'OH', 'NH2', 'O', 'CN']
        if group not in valid:
            print(f"Error: Invalid group '{group}'. Use: {', '.join(valid)}")
            sys.exit(1)
        assignments = {i: group for i in range(len(sites))}
        print(f"\nAuto-passivating all {len(sites)} sites with -{group}")
    else:
        assignments = interactive_menu(sites, atoms)
    
    if not assignments:
        print("\nNo groups assigned. Exiting without changes.")
        return
    
    # Add functional groups
    print("\n" + "=" * 60)
    print("ADDING FUNCTIONAL GROUPS")
    print("=" * 60)
    
    atoms, added = add_functional_groups(atoms, sites, assignments)
    
    print(f"\nAdded {len(added)} atom(s):")
    for info in added:
        print(f"  {info['symbol']:2s} at ({info['position'][0]:7.3f}, {info['position'][1]:7.3f}, {info['position'][2]:7.3f})"
              f" [{info['group']}] → atom {info['attached_to']}")
    
    # Determine output filename
    if args.output:
        output_file = args.output
    else:
        import os
        base = os.path.splitext(args.structure)[0]
        # Add group info to filename
        groups_used = sorted(set(assignments.values()))
        group_str = "_".join(groups_used)
        output_file = f"{base}_passivated_{group_str}.cif"
    
    # Save
    io.write(output_file, atoms)
    print(f"\nSaved: {output_file}")
    print(f"New formula: {atoms.get_chemical_formula()}")
    print(f"Total atoms: {len(atoms)}")
    
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print(f"""
The functional groups have been placed with approximate geometry.
You MUST optimize the structure before running DFT calculations:

  # Quick optimization (draft quality)
  mpirun -n 4 gpaw python -- optimize_geometry.py {output_file} --quality draft

  # Or standard optimization
  mpirun -n 4 gpaw python -- optimize_geometry.py {output_file}

This will relax the added atoms to their equilibrium positions.
""")


if __name__ == "__main__":
    main()

