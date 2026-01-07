#!/usr/bin/env python3
"""
Visualize supercell from CIF file.

Usage:
    python visualize_supercell.py structure.cif                    # View 1x1x1 (original)
    python visualize_supercell.py structure.cif -s 2 2 1           # View 2x2x1 supercell
    python visualize_supercell.py structure.cif -s 3 3 1 -o out.png  # Save to PNG
"""

import argparse
import numpy as np
from ase import io
from ase.visualize import view
from ase.visualize.plot import plot_atoms
import matplotlib.pyplot as plt


def create_supercell(atoms, supercell):
    """Create supercell from atoms."""
    return atoms * supercell


def visualize_interactive(atoms, title="Structure"):
    """Open interactive ASE GUI viewer."""
    print(f"\n{title}")
    print(f"  Atoms: {len(atoms)}")
    print(f"  Cell: {atoms.cell.lengths()}")
    print(f"  Formula: {atoms.get_chemical_formula()}")
    print("\nOpening interactive viewer...")
    print("(Close the viewer window to exit)")
    view(atoms)


def visualize_matplotlib(atoms, output_file, title="Structure", supercell=(1,1,1)):
    """Create publication-quality PNG visualization."""
    
    # Create figure with multiple views
    fig = plt.figure(figsize=(14, 5))
    fig.suptitle(f"{title}\n{atoms.get_chemical_formula()} | "
                 f"Supercell: {supercell[0]}×{supercell[1]}×{supercell[2]} | "
                 f"{len(atoms)} atoms", fontsize=12, fontweight='bold')
    
    # Top view (looking down z-axis)
    ax1 = fig.add_subplot(131)
    plot_atoms(atoms, ax1, rotation='0x,0y,0z', show_unit_cell=2)
    ax1.set_title('Top view (xy plane)', fontsize=10)
    ax1.set_xlabel('x (Å)')
    ax1.set_ylabel('y (Å)')
    ax1.axis('equal')
    
    # Side view (looking down y-axis)
    ax2 = fig.add_subplot(132)
    plot_atoms(atoms, ax2, rotation='90x,0y,0z', show_unit_cell=2)
    ax2.set_title('Side view (xz plane)', fontsize=10)
    ax2.set_xlabel('x (Å)')
    ax2.set_ylabel('z (Å)')
    ax2.axis('equal')
    
    # Perspective view
    ax3 = fig.add_subplot(133)
    plot_atoms(atoms, ax3, rotation='45x,-45y,0z', show_unit_cell=2)
    ax3.set_title('Perspective view', fontsize=10)
    ax3.axis('equal')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"\nSaved visualization to: {output_file}")


def print_structure_info(atoms, supercell):
    """Print detailed structure information."""
    print("\n" + "="*60)
    print("STRUCTURE INFORMATION")
    print("="*60)
    
    print(f"\nSupercell: {supercell[0]} × {supercell[1]} × {supercell[2]}")
    print(f"Formula: {atoms.get_chemical_formula()}")
    print(f"Number of atoms: {len(atoms)}")
    
    print(f"\nCell vectors:")
    for i, vec in enumerate(atoms.cell):
        print(f"  {['a', 'b', 'c'][i]}: [{vec[0]:8.4f}, {vec[1]:8.4f}, {vec[2]:8.4f}] Å")
    
    lengths = atoms.cell.lengths()
    angles = atoms.cell.angles()
    print(f"\nCell parameters:")
    print(f"  a = {lengths[0]:.4f} Å")
    print(f"  b = {lengths[1]:.4f} Å")
    print(f"  c = {lengths[2]:.4f} Å")
    print(f"  α = {angles[0]:.2f}°")
    print(f"  β = {angles[1]:.2f}°")
    print(f"  γ = {angles[2]:.2f}°")
    
    print(f"\nVolume: {atoms.cell.volume:.2f} Å³")
    
    # Element counts
    symbols = atoms.get_chemical_symbols()
    unique_elements = sorted(set(symbols))
    print(f"\nElement counts:")
    for elem in unique_elements:
        count = symbols.count(elem)
        print(f"  {elem}: {count}")
    
    # Z-coordinates range (useful for slabs)
    z_coords = atoms.positions[:, 2]
    print(f"\nZ-coordinate range: {z_coords.min():.3f} to {z_coords.max():.3f} Å")
    print(f"Slab thickness: {z_coords.max() - z_coords.min():.3f} Å")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize CIF structure and supercells",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_supercell.py C3N4_monolayer.cif
  python visualize_supercell.py C3N4_monolayer.cif -s 2 2 1
  python visualize_supercell.py C3N4_monolayer.cif -s 2 2 1 -o supercell.png
  python visualize_supercell.py C3N4_monolayer.cif -s 3 3 1 --no-gui -o supercell.png
        """
    )
    
    parser.add_argument("cif_file", help="Input CIF file")
    parser.add_argument("-s", "--supercell", type=int, nargs=3, default=[1, 1, 1],
                        metavar=("NX", "NY", "NZ"),
                        help="Supercell dimensions (default: 1 1 1)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output PNG file (optional)")
    parser.add_argument("--no-gui", action="store_true",
                        help="Skip interactive viewer (only save PNG)")
    
    args = parser.parse_args()
    
    # Read structure
    print(f"Reading: {args.cif_file}")
    atoms = io.read(args.cif_file, format='cif')
    
    original_natoms = len(atoms)
    supercell = tuple(args.supercell)
    
    # Create supercell
    if supercell != (1, 1, 1):
        atoms = create_supercell(atoms, supercell)
        print(f"Created {supercell[0]}×{supercell[1]}×{supercell[2]} supercell: "
              f"{original_natoms} → {len(atoms)} atoms")
    
    # Print structure info
    print_structure_info(atoms, supercell)
    
    # Save PNG if requested
    if args.output:
        title = args.cif_file.replace('.cif', '').replace('.txt', '')
        visualize_matplotlib(atoms, args.output, title, supercell)
    
    # Open interactive viewer unless --no-gui
    if not args.no_gui:
        visualize_interactive(atoms, f"Supercell {supercell}")


if __name__ == "__main__":
    main()






