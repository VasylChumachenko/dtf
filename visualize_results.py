"""
Visualize and compare initial vs optimized structures.

Usage:
    python visualize_results.py initial.cif optimized.cif
    python visualize_results.py initial.cif optimized.cif --interactive
    python visualize_results.py initial.cif optimized.cif --output comparison.png

Features:
    - Side-by-side structure visualization
    - Cell parameter comparison
    - Bond length analysis
    - Atomic displacement analysis
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ase import io
from ase.geometry.analysis import Analysis
from ase.neighborlist import neighbor_list


def get_bond_lengths(atoms, cutoff=2.0):
    """Get all bond lengths shorter than cutoff."""
    i, j, d = neighbor_list('ijd', atoms, cutoff)
    # Keep only unique pairs (i < j)
    mask = i < j
    return d[mask]


def calculate_displacements(atoms1, atoms2):
    """Calculate atomic displacements between two structures."""
    # Handle periodic boundary conditions
    pos1 = atoms1.get_positions()
    pos2 = atoms2.get_positions()
    cell = atoms1.get_cell()
    
    displacements = []
    for p1, p2 in zip(pos1, pos2):
        # Direct difference
        diff = p2 - p1
        displacements.append(np.linalg.norm(diff))
    
    return np.array(displacements)


def plot_structure_3d(ax, atoms, title, color='C0'):
    """Plot 3D structure on given axis."""
    pos = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    
    # Color by element
    element_colors = {'C': 'gray', 'N': 'blue', 'O': 'red', 'H': 'white', 
                      'S': 'yellow', 'P': 'orange', 'Fe': 'brown'}
    colors = [element_colors.get(s, color) for s in symbols]
    sizes = [100 if s != 'H' else 50 for s in symbols]
    
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=colors, s=sizes, alpha=0.8)
    
    # Draw unit cell
    cell = atoms.get_cell()
    origin = np.array([0, 0, 0])
    for i in range(3):
        ax.plot([origin[0], cell[i, 0]], [origin[1], cell[i, 1]], 
                [origin[2], cell[i, 2]], 'k-', alpha=0.3)
    
    # Draw cell edges
    vertices = [
        origin, cell[0], cell[0] + cell[1], cell[1],
        cell[2], cell[0] + cell[2], cell[0] + cell[1] + cell[2], cell[1] + cell[2]
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
        (4, 5), (5, 6), (6, 7), (7, 4),  # top
        (0, 4), (1, 5), (2, 6), (3, 7)   # vertical
    ]
    for e in edges:
        p1, p2 = vertices[e[0]], vertices[e[1]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'k-', alpha=0.3)
    
    ax.set_xlabel('x (Å)')
    ax.set_ylabel('y (Å)')
    ax.set_zlabel('z (Å)')
    ax.set_title(title)


def print_comparison_table(atoms_init, atoms_opt):
    """Print detailed comparison table."""
    print("\n" + "="*60)
    print("STRUCTURE COMPARISON")
    print("="*60)
    
    # Cell parameters
    lengths_init = atoms_init.cell.lengths()
    lengths_opt = atoms_opt.cell.lengths()
    angles_init = atoms_init.cell.angles()
    angles_opt = atoms_opt.cell.angles()
    vol_init = atoms_init.get_volume()
    vol_opt = atoms_opt.get_volume()
    
    print("\nCell Parameters:")
    print("-"*60)
    print(f"{'Parameter':<12} {'Initial':>15} {'Optimized':>15} {'Change':>15}")
    print("-"*60)
    
    for i, name in enumerate(['a', 'b', 'c']):
        change = lengths_opt[i] - lengths_init[i]
        pct = 100 * change / lengths_init[i]
        print(f"{name:<12} {lengths_init[i]:>15.4f} {lengths_opt[i]:>15.4f} {change:>+10.4f} ({pct:+.2f}%)")
    
    for i, name in enumerate(['α', 'β', 'γ']):
        change = angles_opt[i] - angles_init[i]
        print(f"{name:<12} {angles_init[i]:>15.2f} {angles_opt[i]:>15.2f} {change:>+10.4f}°")
    
    vol_change = vol_opt - vol_init
    vol_pct = 100 * vol_change / vol_init
    print(f"{'Volume (ų)':<12} {vol_init:>15.2f} {vol_opt:>15.2f} {vol_change:>+10.2f} ({vol_pct:+.2f}%)")
    
    # Bond lengths
    bonds_init = get_bond_lengths(atoms_init, cutoff=1.8)
    bonds_opt = get_bond_lengths(atoms_opt, cutoff=1.8)
    
    print("\nBond Length Statistics (cutoff=1.8 Å):")
    print("-"*60)
    print(f"{'Statistic':<12} {'Initial':>15} {'Optimized':>15}")
    print("-"*60)
    print(f"{'Count':<12} {len(bonds_init):>15d} {len(bonds_opt):>15d}")
    if len(bonds_init) > 0 and len(bonds_opt) > 0:
        print(f"{'Min':<12} {bonds_init.min():>15.4f} {bonds_opt.min():>15.4f}")
        print(f"{'Max':<12} {bonds_init.max():>15.4f} {bonds_opt.max():>15.4f}")
        print(f"{'Mean':<12} {bonds_init.mean():>15.4f} {bonds_opt.mean():>15.4f}")
        print(f"{'Std':<12} {bonds_init.std():>15.4f} {bonds_opt.std():>15.4f}")
    
    # Atomic displacements
    if len(atoms_init) == len(atoms_opt):
        displacements = calculate_displacements(atoms_init, atoms_opt)
        print("\nAtomic Displacements:")
        print("-"*60)
        print(f"  Max displacement:  {displacements.max():.4f} Å")
        print(f"  Mean displacement: {displacements.mean():.4f} Å")
        print(f"  Min displacement:  {displacements.min():.4f} Å")
        
        # Find most displaced atoms
        top_indices = np.argsort(displacements)[-3:][::-1]
        print("\n  Most displaced atoms:")
        symbols = atoms_init.get_chemical_symbols()
        for idx in top_indices:
            print(f"    Atom {idx} ({symbols[idx]}): {displacements[idx]:.4f} Å")
    
    print("="*60)


def create_comparison_figure(atoms_init, atoms_opt, output_file=None):
    """Create comparison figure with 3D structures and analysis."""
    fig = plt.figure(figsize=(16, 10))
    
    # 3D structure plots
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    plot_structure_3d(ax1, atoms_init, 'Initial Structure')
    
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    plot_structure_3d(ax2, atoms_opt, 'Optimized Structure')
    
    # Cell parameter comparison (bar chart)
    ax3 = fig.add_subplot(2, 3, 3)
    lengths_init = atoms_init.cell.lengths()
    lengths_opt = atoms_opt.cell.lengths()
    x = np.arange(3)
    width = 0.35
    ax3.bar(x - width/2, lengths_init, width, label='Initial', color='C0', alpha=0.7)
    ax3.bar(x + width/2, lengths_opt, width, label='Optimized', color='C1', alpha=0.7)
    ax3.set_xticks(x)
    ax3.set_xticklabels(['a', 'b', 'c'])
    ax3.set_ylabel('Length (Å)')
    ax3.set_title('Cell Parameters')
    ax3.legend()
    
    # Bond length distribution
    ax4 = fig.add_subplot(2, 3, 4)
    bonds_init = get_bond_lengths(atoms_init, cutoff=1.8)
    bonds_opt = get_bond_lengths(atoms_opt, cutoff=1.8)
    if len(bonds_init) > 0:
        ax4.hist(bonds_init, bins=20, alpha=0.5, label='Initial', color='C0')
    if len(bonds_opt) > 0:
        ax4.hist(bonds_opt, bins=20, alpha=0.5, label='Optimized', color='C1')
    ax4.set_xlabel('Bond Length (Å)')
    ax4.set_ylabel('Count')
    ax4.set_title('Bond Length Distribution')
    ax4.legend()
    
    # Atomic displacements
    ax5 = fig.add_subplot(2, 3, 5)
    if len(atoms_init) == len(atoms_opt):
        displacements = calculate_displacements(atoms_init, atoms_opt)
        ax5.bar(range(len(displacements)), displacements, color='C2', alpha=0.7)
        ax5.axhline(y=displacements.mean(), color='r', linestyle='--', label=f'Mean: {displacements.mean():.3f} Å')
        ax5.set_xlabel('Atom Index')
        ax5.set_ylabel('Displacement (Å)')
        ax5.set_title('Atomic Displacements')
        ax5.legend()
    
    # Volume comparison
    ax6 = fig.add_subplot(2, 3, 6)
    vols = [atoms_init.get_volume(), atoms_opt.get_volume()]
    colors = ['C0', 'C1']
    bars = ax6.bar(['Initial', 'Optimized'], vols, color=colors, alpha=0.7)
    ax6.set_ylabel('Volume (ų)')
    ax6.set_title('Cell Volume')
    # Add value labels
    for bar, vol in zip(bars, vols):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{vol:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {output_file}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Visualize and compare initial vs optimized structures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("initial", help="Initial structure file (CIF, XYZ, etc.)")
    parser.add_argument("optimized", help="Optimized structure file")
    parser.add_argument("-o", "--output", default="comparison.png",
                        help="Output image file (default: comparison.png)")
    parser.add_argument("-i", "--interactive", action="store_true",
                        help="Open interactive ASE viewer")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip plot generation (only print comparison)")
    args = parser.parse_args()
    
    # Read structures
    print(f"Reading initial structure: {args.initial}")
    atoms_init = io.read(args.initial)
    
    print(f"Reading optimized structure: {args.optimized}")
    atoms_opt = io.read(args.optimized)
    
    # Print comparison table
    print_comparison_table(atoms_init, atoms_opt)
    
    # Create figure
    if not args.no_plot:
        fig = create_comparison_figure(atoms_init, atoms_opt, args.output)
        plt.show()
    
    # Interactive viewer
    if args.interactive:
        try:
            from ase.visualize import view
            print("\nOpening interactive viewer...")
            print("  Window 1: Initial structure")
            print("  Window 2: Optimized structure")
            view([atoms_init, atoms_opt])
        except Exception as e:
            print(f"\nCould not open interactive viewer: {e}")
            print("Try: pip install ase[gui]")


if __name__ == "__main__":
    main()

