#!/usr/bin/env python3
"""
Calculate electronic properties from optimized g-C₃N₄ structures.

Properties computed:
    - DOS (Density of States)
    - Band gap (E_g)
    - VBM/CBM positions vs vacuum
    - Total energy
    - Formation energy (for defects)

Usage:
    # Fast draft calculation (~5× speedup)
    gpaw python calculate_properties.py optimized.cif --quality draft
    
    # Standard calculation (default)
    gpaw python calculate_properties.py optimized.cif
    mpirun -n 4 gpaw python -- calculate_properties.py optimized.cif
    
    # Production quality (publication)
    gpaw python calculate_properties.py optimized.cif --quality production
    
    # Formation energy (requires pristine reference)
    gpaw python calculate_properties.py defect.cif --pristine pristine.cif
    
    # High-accuracy HSE06 band gap (expensive!)
    gpaw python calculate_properties.py optimized.cif --hse

Quality presets:
    - draft:      ecut=300, kpts=3×3×1, ~5× faster, ±0.15 eV
    - standard:   ecut=500, kpts=6×6×1, default
    - production: ecut=600, kpts=12×12×1, publication quality

Output files:
    - {name}_dos.png         : DOS plot
    - {name}_dos.json        : DOS data (energies, dos values, band edges)
    - {name}_properties.json : All computed properties
"""

import argparse
import json
import os
import sys
import numpy as np

# Add script directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ase import io
from ase.parallel import parprint, world
from ase.units import kJ, mol

from gpaw import GPAW, PW, FermiDirac

from calc_presets import (
    get_preset, detect_system_type, print_preset_info, 
    get_quality_warning, QUALITY_CHOICES
)


def get_vacuum_level(atoms, calc, direction='z'):
    """
    Get vacuum level from electrostatic potential.
    
    For 2D materials with vacuum along z, this extracts the 
    potential in the vacuum region.
    
    Returns: vacuum level in eV
    """
    try:
        # Get the electrostatic potential
        v = calc.get_electrostatic_potential()
        
        if v is None:
            parprint("Warning: Could not get electrostatic potential")
            return None
        
        # Average in xy plane
        v_z = v.mean(axis=(0, 1))
        
        # Get z coordinates
        cell = atoms.get_cell()
        nz = len(v_z)
        z = np.linspace(0, cell[2, 2], nz, endpoint=False)
        
        # Find vacuum region (where potential is flat/maximum)
        # For slab with vacuum, potential plateaus in vacuum
        v_vacuum = v_z.max()
        
        return v_vacuum
        
    except Exception as e:
        parprint(f"Warning: Vacuum level calculation failed: {e}")
        return None


def calculate_dos(atoms, calc, npts=2001, width=0.1):
    """
    Calculate DOS from converged calculator.
    
    Returns dict with:
        - energies: energy grid (eV)
        - dos: total DOS
        - dos_up/dos_down: spin-polarized if applicable
        - fermi: Fermi level
        - vbm: valence band maximum
        - cbm: conduction band minimum
        - band_gap: band gap
    """
    # Get Fermi level
    fermi = calc.get_fermi_level()
    
    # Collect all eigenvalues from all k-points and spins
    eigenvalues = []
    kpt_weights_list = []
    
    n_kpts = len(calc.get_ibz_k_points())
    n_spins = calc.get_number_of_spins()
    weights = calc.get_k_point_weights()
    
    for k in range(n_kpts):
        for s in range(n_spins):
            try:
                eigs = calc.get_eigenvalues(kpt=k, spin=s)
                w = weights[k]
                for eig in eigs:
                    eigenvalues.append(eig)
                    kpt_weights_list.append(w)
            except:
                pass
    
    eigenvalues = np.array(eigenvalues)
    kpt_weights_list = np.array(kpt_weights_list)
    
    # Create energy grid
    e_min = eigenvalues.min() - 2.0
    e_max = eigenvalues.max() + 2.0
    energies = np.linspace(e_min, e_max, npts)
    
    # Compute DOS with Gaussian broadening
    dos = np.zeros_like(energies)
    for eig, w in zip(eigenvalues, kpt_weights_list):
        dos += w * np.exp(-((energies - eig) / width)**2 / 2)
    dos /= (width * np.sqrt(2 * np.pi))
    
    # Find band edges from eigenvalues directly (more reliable)
    # VBM: highest occupied state (below or at Fermi level)
    # CBM: lowest unoccupied state (above Fermi level)
    occupied = eigenvalues[eigenvalues <= fermi + 0.01]
    unoccupied = eigenvalues[eigenvalues > fermi + 0.01]
    
    if len(occupied) > 0:
        vbm = occupied.max()
    else:
        vbm = fermi
    
    if len(unoccupied) > 0:
        cbm = unoccupied.min()
    else:
        cbm = fermi
    
    # Calculate band gap
    band_gap = cbm - vbm if cbm > vbm else 0.0
    
    result = {
        'energies': energies.tolist(),
        'dos': dos.tolist(),
        'fermi': float(fermi),
        'vbm': float(vbm),
        'cbm': float(cbm),
        'band_gap': float(band_gap),
        'dos_max': float(dos.max()),
        'width': width
    }
    
    return result


def plot_dos(dos_data, output_file, title="Density of States"):
    """Create DOS plot."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        parprint("Warning: matplotlib not available, skipping plot")
        return
    
    energies = np.array(dos_data['energies'])
    dos = np.array(dos_data['dos'])
    fermi = dos_data['fermi']
    vbm = dos_data['vbm']
    cbm = dos_data['cbm']
    band_gap = dos_data['band_gap']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot DOS
    ax.fill_between(energies, dos, alpha=0.3, color='steelblue')
    ax.plot(energies, dos, color='steelblue', linewidth=1)
    
    # Mark Fermi level
    ax.axvline(fermi, color='red', linestyle='--', linewidth=1.5, label=f'Fermi level ({fermi:.2f} eV)')
    
    # Mark band edges
    ax.axvline(vbm, color='green', linestyle=':', linewidth=1.5, label=f'VBM ({vbm:.2f} eV)')
    ax.axvline(cbm, color='orange', linestyle=':', linewidth=1.5, label=f'CBM ({cbm:.2f} eV)')
    
    # Shade band gap region
    if band_gap > 0.1:
        ax.axvspan(vbm, cbm, alpha=0.15, color='yellow', label=f'Gap = {band_gap:.2f} eV')
    
    # Formatting
    ax.set_xlabel('Energy (eV)', fontsize=12)
    ax.set_ylabel('DOS (states/eV)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right')
    ax.set_xlim(fermi - 8, fermi + 8)
    ax.set_ylim(0, dos.max() * 1.1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    parprint(f"Saved DOS plot: {output_file}")


def calculate_formation_energy(E_defect, E_pristine, n_removed, mu_N):
    """
    Calculate defect formation energy.
    
    E_f = E(defect) - E(pristine) + n * μ_N
    
    where:
        E_defect: total energy of defect supercell
        E_pristine: total energy of pristine supercell (same size!)
        n_removed: number of N atoms removed (positive for vacancy)
        mu_N: chemical potential of N
    
    For N-rich conditions: μ_N = E(N₂)/2 = -8.3 eV (approx)
    For N-poor conditions: μ_N ≈ -8.3 + ΔH_f(g-C₃N₄)/4
    """
    E_f = E_defect - E_pristine + n_removed * mu_N
    return E_f


def get_chemical_potential_N2():
    """
    Reference chemical potential for N₂ molecule.
    
    Standard DFT-PBE value: ~-16.6 eV for N₂
    So μ_N = -8.3 eV per N atom
    
    For more accurate calculations, compute E(N₂) explicitly.
    """
    return -8.3  # eV per N atom (N-rich conditions)


def main():
    parser = argparse.ArgumentParser(
        description="Calculate DOS and electronic properties from optimized structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("cif_file", help="Optimized structure (CIF/XYZ)")
    parser.add_argument("--quality", choices=QUALITY_CHOICES, default='standard',
                        help="Quality preset: draft (fast), standard, production (default: standard)")
    parser.add_argument("--ecut", type=float, default=None,
                        help="Plane-wave cutoff in eV (overrides preset)")
    parser.add_argument("--kpts", type=int, nargs=3, default=None,
                        help="k-point grid (overrides preset)")
    parser.add_argument("--smearing", type=float, default=None,
                        help="Fermi-Dirac width in eV (overrides preset)")
    parser.add_argument("--pristine", type=str, default=None,
                        help="Pristine structure for formation energy calculation")
    parser.add_argument("--pristine-energy", type=float, default=None,
                        help="Pre-computed pristine energy (skip calculation)")
    parser.add_argument("--n-removed", type=int, default=1,
                        help="Number of N atoms removed (default: 1)")
    parser.add_argument("--hse", action="store_true",
                        help="Use HSE06 for accurate band gap (expensive!)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip DOS plot generation")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print info only, don't run calculation")
    
    args = parser.parse_args()
    
    # Read structure
    parprint(f"Reading structure: {args.cif_file}")
    atoms = io.read(args.cif_file)
    
    base_name = os.path.splitext(os.path.basename(args.cif_file))[0]
    
    # Detect system type and get preset
    system_type = detect_system_type(atoms)
    preset = get_preset(args.quality, system_type)
    
    # Override with command-line arguments if provided
    ecut = args.ecut if args.ecut is not None else preset['ecut']
    kpts = tuple(args.kpts) if args.kpts is not None else preset['kpts_dos']
    smearing = args.smearing if args.smearing is not None else preset['smearing']
    convergence = preset['convergence']
    dos_width = preset['dos_width']
    dos_npts = preset['dos_npts']
    
    # Print structure info
    parprint(f"\nStructure info:")
    parprint(f"  Formula: {atoms.get_chemical_formula()}")
    parprint(f"  Atoms: {len(atoms)}")
    parprint(f"  Cell: {atoms.cell.lengths()}")
    parprint(f"  System type: {system_type}")
    
    # Print preset info
    print_preset_info(preset, args.quality)
    
    parprint(f"\nEffective parameters:")
    parprint(f"  ecut: {ecut} eV")
    parprint(f"  kpts: {kpts}")
    parprint(f"  smearing: {smearing} eV")
    parprint(f"  convergence: {convergence}")
    parprint(f"  xc: {'HSE06' if args.hse else 'PBE'}")
    
    if args.dry_run:
        parprint("\n[DRY RUN] Exiting.")
        return
    
    # =========================================================
    # Set up calculator
    # =========================================================
    if args.hse:
        parprint("\nUsing HSE06 (this will be slow!)...")
        from gpaw.hybrids import HybridXC
        calc = GPAW(
            mode=PW(ecut),
            xc=HybridXC('HSE06'),
            occupations=FermiDirac(smearing),
            kpts=kpts,
            txt=f"{base_name}_hse_dos.log",
            convergence=convergence
        )
    else:
        calc = GPAW(
            mode=PW(ecut),
            xc="PBE",
            occupations=FermiDirac(smearing),
            kpts=kpts,
            txt=f"{base_name}_dos.log",
            convergence=convergence
        )
    
    atoms.calc = calc
    
    # =========================================================
    # Run SCF and get total energy
    # =========================================================
    parprint("\nRunning SCF calculation...")
    E_total = atoms.get_potential_energy()
    parprint(f"Total energy: {E_total:.6f} eV")
    
    # =========================================================
    # Calculate DOS
    # =========================================================
    parprint("\nCalculating DOS...")
    dos_data = calculate_dos(atoms, calc, npts=dos_npts, width=dos_width)
    dos_data['quality'] = args.quality
    dos_data['uncertainty'] = preset['uncertainty']
    
    parprint(f"\n{'='*50}")
    parprint("ELECTRONIC STRUCTURE")
    parprint(f"{'='*50}")
    parprint(f"  Fermi level:  {dos_data['fermi']:.4f} eV")
    parprint(f"  VBM:          {dos_data['vbm']:.4f} eV")
    parprint(f"  CBM:          {dos_data['cbm']:.4f} eV")
    parprint(f"  Band gap:     {dos_data['band_gap']:.4f} eV")
    
    # =========================================================
    # Vacuum alignment (for 2D materials)
    # =========================================================
    parprint("\nCalculating vacuum level...")
    V_vacuum = get_vacuum_level(atoms, calc)
    
    if V_vacuum is not None:
        # Band edges relative to vacuum
        VBM_vacuum = dos_data['vbm'] - V_vacuum
        CBM_vacuum = dos_data['cbm'] - V_vacuum
        work_function = V_vacuum - dos_data['fermi']
        
        parprint(f"\n{'='*50}")
        parprint("VACUUM ALIGNMENT")
        parprint(f"{'='*50}")
        parprint(f"  Vacuum level: {V_vacuum:.4f} eV")
        parprint(f"  Work function: {work_function:.4f} eV")
        parprint(f"  VBM vs vacuum: {VBM_vacuum:.4f} eV")
        parprint(f"  CBM vs vacuum: {CBM_vacuum:.4f} eV")
        
        # Water redox potentials at pH=0 (vs vacuum)
        # H+/H2: -4.44 eV vs vacuum
        # O2/H2O: -5.67 eV vs vacuum
        H_redox = -4.44
        O_redox = -5.67
        
        parprint(f"\n  Reference potentials (pH=0):")
        parprint(f"    H⁺/H₂:  {H_redox:.2f} eV vs vacuum")
        parprint(f"    O₂/H₂O: {O_redox:.2f} eV vs vacuum")
        
        can_reduce = CBM_vacuum > H_redox
        can_oxidize = VBM_vacuum < O_redox
        parprint(f"\n  HER (CBM > H⁺/H₂): {'✓ YES' if can_reduce else '✗ NO'}")
        parprint(f"  OER (VBM < O₂/H₂O): {'✓ YES' if can_oxidize else '✗ NO'}")
        
        dos_data['vacuum_level'] = float(V_vacuum)
        dos_data['vbm_vacuum'] = float(VBM_vacuum)
        dos_data['cbm_vacuum'] = float(CBM_vacuum)
        dos_data['work_function'] = float(work_function)
    else:
        parprint("  Could not determine vacuum level (no vacuum in cell?)")
    
    # =========================================================
    # Formation energy (if pristine provided)
    # =========================================================
    E_formation = None
    
    if args.pristine_energy is not None:
        E_pristine = args.pristine_energy
        mu_N = get_chemical_potential_N2()
        E_formation = calculate_formation_energy(E_total, E_pristine, args.n_removed, mu_N)
        
        parprint(f"\n{'='*50}")
        parprint("FORMATION ENERGY")
        parprint(f"{'='*50}")
        parprint(f"  E(defect):   {E_total:.6f} eV")
        parprint(f"  E(pristine): {E_pristine:.6f} eV (provided)")
        parprint(f"  μ_N:         {mu_N:.2f} eV (N-rich)")
        parprint(f"  N removed:   {args.n_removed}")
        parprint(f"  E_formation: {E_formation:.4f} eV")
        
    elif args.pristine is not None:
        parprint(f"\nCalculating pristine reference: {args.pristine}")
        pristine_atoms = io.read(args.pristine)
        
        pristine_calc = GPAW(
            mode=PW(args.ecut),
            xc="PBE",
            occupations=FermiDirac(args.smearing),
            kpts=kpts,
            txt=f"{base_name}_pristine.log",
            convergence={"energy": 1e-5, "density": 1e-5}
        )
        pristine_atoms.calc = pristine_calc
        E_pristine = pristine_atoms.get_potential_energy()
        
        mu_N = get_chemical_potential_N2()
        E_formation = calculate_formation_energy(E_total, E_pristine, args.n_removed, mu_N)
        
        parprint(f"\n{'='*50}")
        parprint("FORMATION ENERGY")
        parprint(f"{'='*50}")
        parprint(f"  E(defect):   {E_total:.6f} eV")
        parprint(f"  E(pristine): {E_pristine:.6f} eV")
        parprint(f"  μ_N:         {mu_N:.2f} eV (N-rich conditions)")
        parprint(f"  N removed:   {args.n_removed}")
        parprint(f"  E_formation: {E_formation:.4f} eV")
    
    # =========================================================
    # Save results
    # =========================================================
    
    # Save DOS data
    dos_file = f"{base_name}_dos.json"
    with open(dos_file, 'w') as f:
        json.dump(dos_data, f, indent=2)
    parprint(f"\nSaved DOS data: {dos_file}")
    
    # Compile all properties
    quality_warning = get_quality_warning(args.quality)
    
    properties = {
        'quality': args.quality,
        'quality_warning': quality_warning,
        'uncertainty': preset['uncertainty'],
        'structure': {
            'file': args.cif_file,
            'formula': atoms.get_chemical_formula(),
            'n_atoms': len(atoms),
            'cell_lengths': atoms.cell.lengths().tolist(),
            'cell_angles': atoms.cell.angles().tolist(),
            'system_type': system_type
        },
        'calculation': {
            'ecut': ecut,
            'kpts': list(kpts),
            'xc': 'HSE06' if args.hse else 'PBE',
            'convergence': convergence,
            'smearing': smearing
        },
        'energies': {
            'total_energy_eV': float(E_total),
            'fermi_level_eV': dos_data['fermi']
        },
        'electronic_structure': {
            'band_gap_eV': dos_data['band_gap'],
            'vbm_eV': dos_data['vbm'],
            'cbm_eV': dos_data['cbm']
        }
    }
    
    if V_vacuum is not None:
        properties['vacuum_alignment'] = {
            'vacuum_level_eV': float(V_vacuum),
            'work_function_eV': float(work_function),
            'vbm_vs_vacuum_eV': float(VBM_vacuum),
            'cbm_vs_vacuum_eV': float(CBM_vacuum),
            'can_drive_HER': bool(can_reduce),
            'can_drive_OER': bool(can_oxidize)
        }
    
    if E_formation is not None:
        properties['formation_energy'] = {
            'E_formation_eV': float(E_formation),
            'E_defect_eV': float(E_total),
            'E_pristine_eV': float(E_pristine),
            'mu_N_eV': float(mu_N),
            'n_removed': args.n_removed
        }
    
    props_file = f"{base_name}_properties.json"
    with open(props_file, 'w') as f:
        json.dump(properties, f, indent=2)
    parprint(f"Saved properties: {props_file}")
    
    # Generate DOS plot
    if not args.no_plot:
        plot_file = f"{base_name}_dos.png"
        title = f"DOS: {atoms.get_chemical_formula()} (E_g = {dos_data['band_gap']:.2f} eV)"
        plot_dos(dos_data, plot_file, title=title)
    
    # =========================================================
    # Summary
    # =========================================================
    parprint(f"\n{'='*50}")
    parprint("SUMMARY")
    parprint(f"{'='*50}")
    
    if quality_warning:
        parprint(f"\n  {quality_warning}\n")
    
    parprint(f"  Quality:          {args.quality} ({preset['uncertainty']})")
    parprint(f"  Total energy:     {E_total:.4f} eV")
    parprint(f"  Band gap:         {dos_data['band_gap']:.4f} eV")
    if V_vacuum is not None:
        parprint(f"  CBM vs vacuum:    {CBM_vacuum:.4f} eV")
        parprint(f"  VBM vs vacuum:    {VBM_vacuum:.4f} eV")
    if E_formation is not None:
        parprint(f"  Formation energy: {E_formation:.4f} eV")
    
    parprint(f"\n>>> Next steps:")
    parprint(f"  1. For ΔG_H*: python calculate_h_adsorption.py {args.cif_file} --quality {args.quality}")
    parprint(f"  2. For Bader: Use VASP with LAECHG=.TRUE. and bader analysis")
    if args.quality == 'draft':
        parprint(f"  3. Re-run with --quality standard for publication values")
    parprint(f"\nDone.")


if __name__ == "__main__":
    main()

