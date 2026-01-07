"""
Create 2D slab from bulk layered structure.

Extracts a single layer and adds vacuum for surface calculations (HER, etc.)

Usage:
    python create_2d_slab.py input.cif -o slab.cif --vacuum 20
    python create_2d_slab.py input.cif --layer 0 --vacuum 15
"""

import argparse
import numpy as np
from ase import io, Atoms
from ase.build import make_supercell


def analyze_layers(atoms, tolerance=0.5):
    """Find distinct layers based on z-position."""
    z_pos = atoms.get_positions()[:, 2]
    
    # Cluster z-positions
    layers = []
    sorted_z = sorted(enumerate(z_pos), key=lambda x: x[1])
    
    current_layer = [sorted_z[0][0]]
    current_z = sorted_z[0][1]
    
    for idx, z in sorted_z[1:]:
        if z - current_z < tolerance:
            current_layer.append(idx)
        else:
            layers.append((np.mean([z_pos[i] for i in current_layer]), current_layer))
            current_layer = [idx]
        current_z = z
    layers.append((np.mean([z_pos[i] for i in current_layer]), current_layer))
    
    return sorted(layers, key=lambda x: x[0])


def extract_layer(atoms, layer_indices):
    """Extract atoms from specific layer."""
    symbols = [atoms.get_chemical_symbols()[i] for i in layer_indices]
    positions = atoms.get_positions()[layer_indices]
    
    # Keep original cell (will modify later)
    new_atoms = Atoms(symbols, positions=positions, cell=atoms.cell, pbc=True)
    return new_atoms


def add_vacuum(atoms, vacuum=20.0):
    """Add vacuum along c-axis for 2D slab."""
    # Get current positions
    pos = atoms.get_positions()
    
    # Center slab in z
    z_min = pos[:, 2].min()
    z_max = pos[:, 2].max()
    z_center = (z_min + z_max) / 2
    thickness = z_max - z_min
    
    # Move to center of new cell
    new_c = thickness + vacuum
    pos[:, 2] = pos[:, 2] - z_center + new_c / 2
    
    # Create new cell with vacuum
    cell = atoms.cell.copy()
    cell[2, 2] = new_c
    
    new_atoms = Atoms(
        atoms.get_chemical_symbols(),
        positions=pos,
        cell=cell,
        pbc=[True, True, False]  # No PBC along z for true 2D
    )
    
    return new_atoms


def main():
    parser = argparse.ArgumentParser(
        description="Create 2D slab from bulk layered structure",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", help="Input structure file (CIF, etc.)")
    parser.add_argument("-o", "--output", default="slab.cif",
                        help="Output file (default: slab.cif)")
    parser.add_argument("--vacuum", type=float, default=20.0,
                        help="Vacuum thickness in Å (default: 20)")
    parser.add_argument("--layer", type=int, default=0,
                        help="Which layer to extract (0=bottom, 1=next, etc.)")
    parser.add_argument("--all-layers", action="store_true",
                        help="Keep all layers (just add vacuum)")
    parser.add_argument("--info", action="store_true",
                        help="Only show layer info, don't create slab")
    args = parser.parse_args()
    
    # Read structure
    print(f"Reading: {args.input}")
    atoms = io.read(args.input)
    
    print(f"\nOriginal structure:")
    print(f"  Formula: {atoms.get_chemical_formula()}")
    print(f"  Atoms: {len(atoms)}")
    print(f"  Cell: a={atoms.cell.lengths()[0]:.3f}, b={atoms.cell.lengths()[1]:.3f}, c={atoms.cell.lengths()[2]:.3f} Å")
    
    # Analyze layers
    layers = analyze_layers(atoms)
    print(f"\nDetected {len(layers)} layer(s):")
    for i, (z_avg, indices) in enumerate(layers):
        symbols = [atoms.get_chemical_symbols()[j] for j in indices]
        formula = ''.join(sorted(set(symbols)))
        print(f"  Layer {i}: z ≈ {z_avg:.2f} Å, {len(indices)} atoms ({formula})")
    
    if args.info:
        return
    
    # Create slab
    if args.all_layers:
        print(f"\nKeeping all layers, adding {args.vacuum} Å vacuum...")
        slab = add_vacuum(atoms, args.vacuum)
    else:
        if args.layer >= len(layers):
            print(f"Error: Layer {args.layer} doesn't exist (max: {len(layers)-1})")
            return
        
        print(f"\nExtracting layer {args.layer}...")
        z_avg, indices = layers[args.layer]
        slab = extract_layer(atoms, indices)
        
        print(f"Adding {args.vacuum} Å vacuum...")
        slab = add_vacuum(slab, args.vacuum)
    
    print(f"\n2D Slab created:")
    print(f"  Formula: {slab.get_chemical_formula()}")
    print(f"  Atoms: {len(slab)}")
    print(f"  Cell: a={slab.cell.lengths()[0]:.3f}, b={slab.cell.lengths()[1]:.3f}, c={slab.cell.lengths()[2]:.3f} Å")
    
    z_pos = slab.get_positions()[:, 2]
    print(f"  Slab thickness: {z_pos.max() - z_pos.min():.3f} Å")
    print(f"  Vacuum: {args.vacuum:.1f} Å")
    
    # Save
    io.write(args.output, slab)
    print(f"\nSaved to: {args.output}")
    
    print(f"\n💡 For optimization, use:")
    print(f"   ./run_optimization.sh -n 4 {args.output} --fix-c --kpts 6 6 1")


if __name__ == "__main__":
    main()






