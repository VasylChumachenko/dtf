#!/usr/bin/env python3
"""
Calculate hydrogen adsorption free energy (ΔG_H*) for HER activity.

The Gibbs free energy of H adsorption is:
    ΔG_H* = E(surface+H) - E(surface) - 0.5*E(H₂) + ΔZPE - TΔS

For computational screening, using standard corrections:
    ΔG_H* = ΔE_H + 0.24 eV  (PBE, approximate)

where ΔE_H = E(surface+H) - E(surface) - 0.5*E(H₂)

Optimal HER catalyst: ΔG_H* ≈ 0 eV (Sabatier principle)

Usage:
    # Fast draft calculation
    python calculate_h_adsorption.py structure.cif --site-index 15 --quality draft
    
    # Find adsorption sites and add H
    python calculate_h_adsorption.py structure.cif --find-sites
    
    # Calculate ΔG_H* for H at specific position  
    python calculate_h_adsorption.py structure.cif --add-h 5.0 3.0 12.5
    
    # Use pre-computed surface energy (saves time!)
    python calculate_h_adsorption.py structure.cif --add-h 5.0 3.0 12.5 --surface-energy -450.123

Quality presets:
    - draft:      Loose H optimization, faster SCF (~3× speedup)
    - standard:   Normal workflow
    - production: Tight convergence

Reference:
    - Nørskov et al., J. Electrochem. Soc. 152, J23 (2005)
    - ΔG_H* = ΔE_H + 0.24 eV is a common approximation
"""

import argparse
import json
import os
import sys
import numpy as np

# Add script directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ase import io, Atom
from ase.parallel import parprint, world
from ase.neighborlist import neighbor_list

from gpaw import GPAW, PW, FermiDirac

from calc_presets import (
    get_preset, detect_system_type, print_preset_info,
    get_quality_warning, QUALITY_CHOICES
)


# Standard corrections for ΔG_H* calculation (PBE)
# ZPE(H*) - 0.5*ZPE(H2) ≈ 0.04 eV
# -TΔS ≈ 0.20 eV at 298K (from loss of translational entropy)
# Total correction: ~0.24 eV
DELTA_G_CORRECTION = 0.24  # eV

# H2 reference energy (PBE, computed)
# E(H2) ≈ -6.77 eV (depends on your setup - should be computed!)
E_H2_REFERENCE = -6.77  # eV


def get_h2_energy(ecut=500, box_size=15.0):
    """
    Calculate H₂ molecule energy in vacuum.
    
    Returns E(H₂) in eV.
    
    Run this once to get your reference!
    """
    from ase import Atoms
    from ase.optimize import BFGS
    
    # H2 in a box
    h2 = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])  # ~0.74 Å bond
    h2.center(vacuum=box_size/2)
    
    calc = GPAW(
        mode=PW(ecut),
        xc='PBE',
        txt='h2_reference.log',
        convergence={'energy': 1e-6}
    )
    h2.calc = calc
    
    # Optimize H2
    opt = BFGS(h2, logfile='h2_opt.log')
    opt.run(fmax=0.01)
    
    E_h2 = h2.get_potential_energy()
    d_HH = h2.get_distance(0, 1)
    
    parprint(f"H₂ reference:")
    parprint(f"  E(H₂) = {E_h2:.6f} eV")
    parprint(f"  d(H-H) = {d_HH:.4f} Å")
    
    return E_h2


def find_adsorption_sites(atoms, defect_type='vacancy'):
    """
    Find potential H adsorption sites on g-C₃N₄.
    
    For N-vacancy: H adsorbs on neighboring C atoms
    For pristine: H adsorbs on N sites (weak) or C sites
    For doped: H adsorbs on/near dopant
    
    Returns list of (site_index, position, site_type)
    """
    symbols = atoms.get_chemical_symbols()
    positions = atoms.positions
    
    # Get coordination
    i_list, j_list, d_list = neighbor_list('ijd', atoms, cutoff=1.8)
    coordination = np.bincount(i_list, minlength=len(atoms))
    
    sites = []
    
    # Find undercoordinated atoms (near vacancy)
    for idx, (sym, coord) in enumerate(zip(symbols, coordination)):
        if sym == 'C' and coord < 3:  # Undercoordinated C (normal is 3)
            pos = positions[idx]
            # H position slightly above the C atom
            h_pos = pos + np.array([0, 0, 1.1])  # 1.1 Å above
            sites.append({
                'atom_index': idx,
                'atom_symbol': sym,
                'coordination': int(coord),
                'position': pos.tolist(),
                'h_position': h_pos.tolist(),
                'site_type': 'undercoordinated_C'
            })
        
        elif sym == 'N' and coord == 2:  # Bridging N (possible site)
            pos = positions[idx]
            h_pos = pos + np.array([0, 0, 1.05])
            sites.append({
                'atom_index': idx,
                'atom_symbol': sym,
                'coordination': int(coord),
                'position': pos.tolist(),
                'h_position': h_pos.tolist(),
                'site_type': 'bridging_N'
            })
    
    # Also check for dopant atoms (S, P, etc.)
    for idx, sym in enumerate(symbols):
        if sym in ['S', 'P', 'B', 'O']:
            pos = positions[idx]
            h_pos = pos + np.array([0, 0, 1.2])
            sites.append({
                'atom_index': idx,
                'atom_symbol': sym,
                'coordination': int(coordination[idx]),
                'position': pos.tolist(),
                'h_position': h_pos.tolist(),
                'site_type': f'{sym}_dopant'
            })
    
    return sites


def add_hydrogen(atoms, position, optimize_h=True):
    """
    Add H atom at specified position.
    
    Args:
        atoms: ASE Atoms object
        position: [x, y, z] for H atom
        optimize_h: If True, do quick local optimization
    
    Returns new Atoms object with H added.
    """
    new_atoms = atoms.copy()
    new_atoms.append(Atom('H', position))
    
    return new_atoms


def calculate_delta_E_H(E_surf_H, E_surf, E_H2=None):
    """
    Calculate hydrogen adsorption energy.
    
    ΔE_H = E(surface+H) - E(surface) - 0.5*E(H₂)
    """
    if E_H2 is None:
        E_H2 = E_H2_REFERENCE
    
    delta_E = E_surf_H - E_surf - 0.5 * E_H2
    return delta_E


def calculate_delta_G_H(delta_E_H, correction=DELTA_G_CORRECTION):
    """
    Calculate Gibbs free energy of H adsorption.
    
    ΔG_H* = ΔE_H + correction
    
    Standard correction includes ZPE and entropy terms.
    """
    return delta_E_H + correction


def main():
    parser = argparse.ArgumentParser(
        description="Calculate H adsorption free energy (ΔG_H*)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("cif_file", help="Input structure (CIF/XYZ)")
    parser.add_argument("--quality", choices=QUALITY_CHOICES, default='standard',
                        help="Quality preset: draft (fast), standard, production (default: standard)")
    parser.add_argument("--find-sites", action="store_true",
                        help="Find and print potential adsorption sites")
    parser.add_argument("--add-h", type=float, nargs=3, metavar=('X', 'Y', 'Z'),
                        help="Add H at position and calculate energy")
    parser.add_argument("--site-index", type=int, default=None,
                        help="Add H near atom at this index (1.1 Å above)")
    parser.add_argument("--surface-energy", type=float, default=None,
                        help="Pre-computed surface energy (skip calculation)")
    parser.add_argument("--h2-energy", type=float, default=E_H2_REFERENCE,
                        help=f"H₂ reference energy (default: {E_H2_REFERENCE:.2f} eV)")
    parser.add_argument("--calc-h2", action="store_true",
                        help="Calculate H₂ reference energy")
    parser.add_argument("--ecut", type=float, default=None,
                        help="Plane-wave cutoff in eV (overrides preset)")
    parser.add_argument("--kpts", type=int, nargs=3, default=None,
                        help="k-point grid (overrides preset)")
    parser.add_argument("--fmax", type=float, default=None,
                        help="Force convergence for H optimization (overrides preset)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print info only, don't run calculation")
    
    args = parser.parse_args()
    
    # Read structure first to detect system type
    parprint(f"Reading structure: {args.cif_file}")
    atoms = io.read(args.cif_file)
    base_name = os.path.splitext(os.path.basename(args.cif_file))[0]
    
    # Detect system type and get preset
    system_type = detect_system_type(atoms)
    preset = get_preset(args.quality, system_type)
    
    # Override with command-line arguments if provided
    ecut = args.ecut if args.ecut is not None else preset['ecut']
    kpts = tuple(args.kpts) if args.kpts is not None else preset['kpts']
    fmax = args.fmax if args.fmax is not None else preset['h_opt_fmax']
    h_opt_steps = preset['h_opt_steps']
    convergence = preset['convergence']
    smearing = preset['smearing']
    
    # Calculate H2 reference
    if args.calc_h2:
        parprint("Calculating H₂ reference energy...")
        E_H2 = get_h2_energy(ecut=ecut)
        parprint(f"\nUse: --h2-energy {E_H2:.6f}")
        return
    
    E_H2 = args.h2_energy
    
    parprint(f"\nStructure: {atoms.get_chemical_formula()}")
    parprint(f"Atoms: {len(atoms)}")
    parprint(f"System type: {system_type}")
    
    # Print preset info
    print_preset_info(preset, args.quality)
    
    # Find adsorption sites
    if args.find_sites:
        parprint("\n" + "="*60)
        parprint("POTENTIAL H ADSORPTION SITES")
        parprint("="*60)
        
        sites = find_adsorption_sites(atoms)
        
        if not sites:
            parprint("\nNo obvious adsorption sites found.")
            parprint("Try adding H manually with --add-h X Y Z")
        else:
            parprint(f"\nFound {len(sites)} potential sites:\n")
            for i, site in enumerate(sites):
                parprint(f"  Site {i+1}:")
                parprint(f"    Atom: {site['atom_symbol']} (index {site['atom_index']})")
                parprint(f"    Coordination: {site['coordination']}")
                parprint(f"    Type: {site['site_type']}")
                parprint(f"    H position: ({site['h_position'][0]:.3f}, {site['h_position'][1]:.3f}, {site['h_position'][2]:.3f})")
                parprint()
        
        # Save sites
        sites_file = f"{base_name}_h_sites.json"
        with open(sites_file, 'w') as f:
            json.dump(sites, f, indent=2)
        parprint(f"Saved sites to: {sites_file}")
        
        parprint("\n>>> To calculate ΔG_H*:")
        if sites:
            h_pos = sites[0]['h_position']
            parprint(f"  python calculate_h_adsorption.py {args.cif_file} --add-h {h_pos[0]:.3f} {h_pos[1]:.3f} {h_pos[2]:.3f}")
        return
    
    # Determine H position
    h_position = None
    
    if args.add_h is not None:
        h_position = np.array(args.add_h)
    elif args.site_index is not None:
        pos = atoms.positions[args.site_index]
        h_position = pos + np.array([0, 0, 1.1])
        parprint(f"Adding H above atom {args.site_index} at {h_position}")
    else:
        parprint("ERROR: Specify H position with --add-h or --site-index")
        parprint("       Use --find-sites to see potential sites")
        return
    
    parprint(f"\nH position: ({h_position[0]:.3f}, {h_position[1]:.3f}, {h_position[2]:.3f})")
    
    parprint(f"\nEffective parameters:")
    parprint(f"  ecut: {ecut} eV")
    parprint(f"  kpts: {kpts}")
    parprint(f"  fmax: {fmax} eV/Å")
    parprint(f"  H opt steps: {h_opt_steps}")
    
    if args.dry_run:
        parprint("\n[DRY RUN] Exiting.")
        return
    
    # =========================================================
    # Calculate surface energy (if not provided)
    # =========================================================
    if args.surface_energy is not None:
        E_surf = args.surface_energy
        parprint(f"\nUsing provided surface energy: {E_surf:.6f} eV")
    else:
        parprint("\nCalculating surface energy...")
        calc_surf = GPAW(
            mode=PW(ecut),
            xc='PBE',
            occupations=FermiDirac(smearing),
            kpts=kpts,
            txt=f"{base_name}_surface.log",
            convergence=convergence
        )
        atoms.calc = calc_surf
        E_surf = atoms.get_potential_energy()
        parprint(f"E(surface) = {E_surf:.6f} eV")
    
    # =========================================================
    # Add H and optimize
    # =========================================================
    parprint("\nAdding H atom...")
    atoms_H = add_hydrogen(atoms, h_position)
    
    # Save structure with H
    h_structure_file = f"{base_name}_with_H.cif"
    io.write(h_structure_file, atoms_H)
    parprint(f"Saved: {h_structure_file}")
    
    # Calculate energy with H
    parprint("\nCalculating energy with H adsorbed...")
    calc_H = GPAW(
        mode=PW(ecut),
        xc='PBE',
        occupations=FermiDirac(smearing),
        kpts=kpts,
        txt=f"{base_name}_with_H.log",
        convergence=convergence
    )
    atoms_H.calc = calc_H
    
    # Optimize H position (freeze surface)
    parprint(f"Optimizing H position (fmax={fmax}, max_steps={h_opt_steps})...")
    from ase.optimize import BFGS
    from ase.constraints import FixAtoms
    
    # Fix all atoms except H (last atom)
    constraint = FixAtoms(indices=list(range(len(atoms_H) - 1)))
    atoms_H.set_constraint(constraint)
    
    opt = BFGS(atoms_H, trajectory=f"{base_name}_H_opt.traj", logfile=f"{base_name}_H_opt.log")
    opt.run(fmax=fmax, steps=h_opt_steps)
    
    E_surf_H = atoms_H.get_potential_energy()
    parprint(f"E(surface+H) = {E_surf_H:.6f} eV")
    
    # Get final H position
    h_final = atoms_H.positions[-1]
    parprint(f"Final H position: ({h_final[0]:.3f}, {h_final[1]:.3f}, {h_final[2]:.3f})")
    
    # Save optimized structure
    io.write(f"{base_name}_with_H_opt.cif", atoms_H)
    
    # =========================================================
    # Calculate ΔG_H*
    # =========================================================
    delta_E_H = calculate_delta_E_H(E_surf_H, E_surf, E_H2)
    delta_G_H = calculate_delta_G_H(delta_E_H)
    
    parprint(f"\n{'='*60}")
    parprint("HYDROGEN ADSORPTION RESULTS")
    parprint(f"{'='*60}")
    parprint(f"  E(surface):     {E_surf:.6f} eV")
    parprint(f"  E(surface+H):   {E_surf_H:.6f} eV")
    parprint(f"  E(H₂) ref:      {E_H2:.6f} eV")
    parprint(f"  0.5·E(H₂):      {0.5*E_H2:.6f} eV")
    parprint(f"\n  ΔE_H:           {delta_E_H:.4f} eV")
    parprint(f"  Correction:     {DELTA_G_CORRECTION:.4f} eV (ZPE + entropy)")
    parprint(f"\n  ΔG_H*:          {delta_G_H:.4f} eV")
    
    # Interpret result
    parprint(f"\n{'='*60}")
    parprint("HER ACTIVITY ASSESSMENT")
    parprint(f"{'='*60}")
    
    if abs(delta_G_H) < 0.1:
        parprint(f"  |ΔG_H*| = {abs(delta_G_H):.3f} eV < 0.1 eV")
        parprint("  >>> EXCELLENT HER activity (near thermoneutral)")
    elif abs(delta_G_H) < 0.2:
        parprint(f"  |ΔG_H*| = {abs(delta_G_H):.3f} eV < 0.2 eV")
        parprint("  >>> GOOD HER activity")
    elif abs(delta_G_H) < 0.5:
        parprint(f"  |ΔG_H*| = {abs(delta_G_H):.3f} eV < 0.5 eV")
        parprint("  >>> MODERATE HER activity")
    else:
        parprint(f"  |ΔG_H*| = {abs(delta_G_H):.3f} eV > 0.5 eV")
        parprint("  >>> POOR HER activity (too strong/weak H binding)")
    
    if delta_G_H < 0:
        parprint(f"  H binding: TOO STRONG (hard to release H₂)")
    else:
        parprint(f"  H binding: TOO WEAK (hard to adsorb H⁺)")
    
    # =========================================================
    # Save results
    # =========================================================
    quality_warning = get_quality_warning(args.quality)
    
    results = {
        'quality': args.quality,
        'quality_warning': quality_warning,
        'uncertainty': preset['uncertainty'],
        'structure': args.cif_file,
        'h_position_initial': h_position.tolist(),
        'h_position_final': h_final.tolist(),
        'energies': {
            'E_surface_eV': float(E_surf),
            'E_surface_H_eV': float(E_surf_H),
            'E_H2_ref_eV': float(E_H2)
        },
        'adsorption': {
            'delta_E_H_eV': float(delta_E_H),
            'delta_G_H_eV': float(delta_G_H),
            'zpe_entropy_correction_eV': DELTA_G_CORRECTION
        },
        'calculation': {
            'ecut': ecut,
            'kpts': list(kpts),
            'fmax': fmax,
            'h_opt_steps': h_opt_steps
        }
    }
    
    results_file = f"{base_name}_h_adsorption.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    parprint(f"\nSaved results: {results_file}")
    
    parprint(f"\n>>> Summary:")
    if quality_warning:
        parprint(f"  {quality_warning}")
    parprint(f"  Quality:  {args.quality} ({preset['uncertainty']})")
    parprint(f"  ΔG_H* = {delta_G_H:.4f} eV")
    parprint(f"  Target: |ΔG_H*| < 0.1 eV for optimal HER")
    if args.quality == 'draft':
        parprint(f"\n  Re-run with --quality standard for publication values")
    parprint("\nDone.")


if __name__ == "__main__":
    main()

