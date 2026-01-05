#!/usr/bin/env python3
"""
Complete DFT analysis workflow for g-C₃N₄ photocatalyst screening.

This script orchestrates the full workflow:
    1. Geometry optimization (if not already done)
    2. DOS calculation → Band gap, VBM, CBM
    3. Vacuum alignment → Band edge positions vs vacuum
    4. H adsorption → ΔG_H* for HER activity
    5. Formation energy (for defects)
    6. Charge analysis → Bader/Hirshfeld charges

Usage:
    # Fast draft analysis (~5× speedup)
    python run_full_analysis.py structure.cif --quality draft
    
    # Full analysis of optimized defect structure
    python run_full_analysis.py defect_optimized.cif --pristine pristine_optimized.cif
    
    # Quick DOS-only analysis
    python run_full_analysis.py structure.cif --dos-only
    
    # With H adsorption (specify site)
    python run_full_analysis.py structure.cif --h-site 5

Quality presets:
    - draft:      ~5× faster, ±0.15 eV accuracy (screening)
    - standard:   Normal workflow (default)
    - production: Publication quality

Output:
    - {name}_summary.json     : All properties in one file
    - {name}_dos.png          : DOS plot
    - {name}_report.txt       : Human-readable report
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

# Add script directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from ase import io
from ase.parallel import parprint, world

from calc_presets import (
    get_preset, detect_system_type, print_preset_info,
    get_quality_warning, QUALITY_CHOICES
)


def run_command(cmd, description=""):
    """Run a subprocess command."""
    parprint(f"\n>>> {description}")
    parprint(f"    Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        parprint(f"    ERROR: {result.stderr}")
        return False
    
    return True


def load_json(filename):
    """Load JSON file if it exists."""
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return None


def generate_report(summary, output_file):
    """Generate human-readable report."""
    lines = []
    lines.append("=" * 70)
    lines.append("DFT ANALYSIS REPORT: g-C₃N₄ PHOTOCATALYST")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)
    
    # Quality warning
    quality = summary.get('quality', 'standard')
    if quality == 'draft':
        lines.append("")
        lines.append("⚠️  DRAFT MODE - Results are approximate!")
        lines.append(f"    Uncertainty: {summary.get('uncertainty', '±0.15 eV')}")
        lines.append("    Re-run with --quality standard for publication values.")
        lines.append("")
    elif quality == 'production':
        lines.append("")
        lines.append("✓ PRODUCTION MODE - Publication quality settings.")
        lines.append("")
    
    # Structure info
    if 'structure' in summary:
        s = summary['structure']
        lines.append("\n## STRUCTURE")
        lines.append(f"   File:    {s.get('file', 'N/A')}")
        lines.append(f"   Formula: {s.get('formula', 'N/A')}")
        lines.append(f"   Atoms:   {s.get('n_atoms', 'N/A')}")
        if 'defect_type' in s:
            lines.append(f"   Defect:  {s['defect_type']}")
    
    # Electronic structure
    if 'electronic_structure' in summary:
        es = summary['electronic_structure']
        lines.append("\n## ELECTRONIC STRUCTURE")
        lines.append(f"   Band gap (E_g):    {es.get('band_gap_eV', 'N/A'):.3f} eV")
        lines.append(f"   VBM:               {es.get('vbm_eV', 'N/A'):.3f} eV")
        lines.append(f"   CBM:               {es.get('cbm_eV', 'N/A'):.3f} eV")
        lines.append(f"   Fermi level:       {es.get('fermi_level_eV', 'N/A'):.3f} eV")
    
    # Vacuum alignment
    if 'vacuum_alignment' in summary:
        va = summary['vacuum_alignment']
        lines.append("\n## BAND ALIGNMENT (vs vacuum)")
        lines.append(f"   Vacuum level:      {va.get('vacuum_level_eV', 'N/A'):.3f} eV")
        lines.append(f"   VBM vs vacuum:     {va.get('vbm_vs_vacuum_eV', 'N/A'):.3f} eV")
        lines.append(f"   CBM vs vacuum:     {va.get('cbm_vs_vacuum_eV', 'N/A'):.3f} eV")
        lines.append(f"   Work function:     {va.get('work_function_eV', 'N/A'):.3f} eV")
        lines.append("")
        lines.append(f"   Can drive HER:     {'✓ YES' if va.get('can_drive_HER') else '✗ NO'}")
        lines.append(f"   Can drive OER:     {'✓ YES' if va.get('can_drive_OER') else '✗ NO'}")
    
    # HER activity
    if 'her_activity' in summary:
        her = summary['her_activity']
        dg = her.get('delta_G_H_eV', 0)
        lines.append("\n## HER ACTIVITY (ΔG_H*)")
        lines.append(f"   ΔE_H:              {her.get('delta_E_H_eV', 'N/A'):.4f} eV")
        lines.append(f"   ΔG_H*:             {dg:.4f} eV")
        lines.append(f"   |ΔG_H*|:           {abs(dg):.4f} eV")
        
        # Rating
        if abs(dg) < 0.1:
            rating = "EXCELLENT (near thermoneutral)"
        elif abs(dg) < 0.2:
            rating = "GOOD"
        elif abs(dg) < 0.5:
            rating = "MODERATE"
        else:
            rating = "POOR"
        lines.append(f"   Activity rating:   {rating}")
        lines.append(f"   H binding:         {'Too strong' if dg < 0 else 'Too weak'}")
    
    # Formation energy
    if 'formation_energy' in summary:
        fe = summary['formation_energy']
        lines.append("\n## DEFECT FORMATION ENERGY")
        lines.append(f"   E_formation:       {fe.get('E_formation_eV', 'N/A'):.4f} eV")
        lines.append(f"   E(defect):         {fe.get('E_defect_eV', 'N/A'):.4f} eV")
        lines.append(f"   E(pristine):       {fe.get('E_pristine_eV', 'N/A'):.4f} eV")
        lines.append(f"   μ_N (N-rich):      {fe.get('mu_N_eV', 'N/A'):.2f} eV")
    
    # Charges
    if 'charge_analysis' in summary:
        ca = summary['charge_analysis']
        lines.append("\n## CHARGE ANALYSIS")
        lines.append(f"   Method:            {ca.get('method', 'N/A')}")
        if 'defect_site_charge' in ca:
            lines.append(f"   Defect site:       {ca['defect_site_charge']:.4f} e")
        if 'by_element' in ca:
            for sym, data in ca['by_element'].items():
                lines.append(f"   {sym}: mean={data['mean']:.3f}e, std={data['std']:.3f}e")
    
    # Energies summary
    if 'energies' in summary:
        en = summary['energies']
        lines.append("\n## ENERGIES")
        lines.append(f"   Total energy:      {en.get('total_energy_eV', 'N/A'):.6f} eV")
    
    # Final summary table (for literature comparison)
    lines.append("\n" + "=" * 70)
    lines.append("SUMMARY TABLE (for literature comparison)")
    lines.append("=" * 70)
    lines.append(f"{'Property':<25} {'Value':<15} {'Unit'}")
    lines.append("-" * 70)
    
    if 'electronic_structure' in summary:
        es = summary['electronic_structure']
        lines.append(f"{'Band gap (E_g)':<25} {es.get('band_gap_eV', 'N/A'):<15.3f} eV")
    
    if 'vacuum_alignment' in summary:
        va = summary['vacuum_alignment']
        lines.append(f"{'CBM vs vacuum':<25} {va.get('cbm_vs_vacuum_eV', 'N/A'):<15.3f} eV")
        lines.append(f"{'VBM vs vacuum':<25} {va.get('vbm_vs_vacuum_eV', 'N/A'):<15.3f} eV")
    
    if 'her_activity' in summary:
        her = summary['her_activity']
        lines.append(f"{'ΔG_H*':<25} {her.get('delta_G_H_eV', 'N/A'):<15.4f} eV")
    
    if 'formation_energy' in summary:
        fe = summary['formation_energy']
        lines.append(f"{'E_formation':<25} {fe.get('E_formation_eV', 'N/A'):<15.4f} eV")
    
    if 'charge_analysis' in summary and 'defect_site_charge' in summary['charge_analysis']:
        ca = summary['charge_analysis']
        lines.append(f"{'Bader charge (defect)':<25} {ca.get('defect_site_charge', 'N/A'):<15.4f} e")
    
    lines.append("=" * 70)
    
    report = "\n".join(lines)
    
    with open(output_file, 'w') as f:
        f.write(report)
    
    parprint(f"\nSaved report: {output_file}")
    parprint(report)
    
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Complete DFT analysis workflow for g-C₃N₄",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("cif_file", help="Optimized structure (CIF/XYZ)")
    parser.add_argument("--quality", choices=QUALITY_CHOICES, default='standard',
                        help="Quality preset: draft (fast), standard, production (default: standard)")
    parser.add_argument("--pristine", type=str, default=None,
                        help="Pristine structure for formation energy")
    parser.add_argument("--h-site", type=int, default=None,
                        help="Atom index for H adsorption site")
    parser.add_argument("--dos-only", action="store_true",
                        help="Only calculate DOS (skip H adsorption)")
    parser.add_argument("--skip-charges", action="store_true",
                        help="Skip charge analysis")
    parser.add_argument("--ecut", type=float, default=None,
                        help="Plane-wave cutoff (overrides preset)")
    parser.add_argument("--kpts", type=int, nargs=3, default=None,
                        help="k-point grid (overrides preset)")
    parser.add_argument("--defect-type", type=str, default=None,
                        help="Defect type label (e.g., 'N_vacancy_bridging')")
    
    args = parser.parse_args()
    
    # Read structure
    parprint(f"\n{'='*70}")
    parprint("DFT ANALYSIS WORKFLOW")
    parprint(f"{'='*70}")
    
    atoms = io.read(args.cif_file)
    base_name = os.path.splitext(os.path.basename(args.cif_file))[0]
    
    # Detect system type and get preset
    system_type = detect_system_type(atoms)
    preset = get_preset(args.quality, system_type)
    
    # Override with command-line arguments if provided
    ecut = args.ecut if args.ecut is not None else preset['ecut']
    kpts_dos = tuple(args.kpts) if args.kpts is not None else preset['kpts_dos']
    kpts_scf = tuple(args.kpts) if args.kpts is not None else preset['kpts']
    convergence = preset['convergence']
    smearing = preset['smearing']
    dos_width = preset['dos_width']
    dos_npts = preset['dos_npts']
    h_opt_fmax = preset['h_opt_fmax']
    h_opt_steps = preset['h_opt_steps']
    
    parprint(f"Structure: {args.cif_file}")
    parprint(f"Formula:   {atoms.get_chemical_formula()}")
    parprint(f"Atoms:     {len(atoms)}")
    parprint(f"System:    {system_type}")
    
    # Print preset info
    print_preset_info(preset, args.quality)
    
    quality_warning = get_quality_warning(args.quality)
    
    # Initialize summary
    summary = {
        'quality': args.quality,
        'quality_warning': quality_warning,
        'uncertainty': preset['uncertainty'],
        'structure': {
            'file': args.cif_file,
            'formula': atoms.get_chemical_formula(),
            'n_atoms': len(atoms),
            'cell_lengths': atoms.cell.lengths().tolist(),
            'defect_type': args.defect_type,
            'system_type': system_type
        },
        'calculation': {
            'ecut': ecut,
            'kpts_dos': list(kpts_dos),
            'kpts_scf': list(kpts_scf),
            'xc': 'PBE',
            'convergence': convergence
        }
    }
    
    # =========================================================
    # Step 1: DOS and Electronic Structure
    # =========================================================
    parprint(f"\n{'='*70}")
    parprint("STEP 1: DOS and Electronic Structure")
    parprint(f"{'='*70}")
    
    parprint(f"\nUsing parameters:")
    parprint(f"  ecut: {ecut} eV")
    parprint(f"  kpts: {kpts_dos}")
    parprint(f"  convergence: {convergence}")
    
    # Import and run DOS calculation inline (more control)
    from gpaw import GPAW, PW, FermiDirac
    
    calc = GPAW(
        mode=PW(ecut),
        xc='PBE',
        occupations=FermiDirac(smearing),
        kpts=kpts_dos,
        txt=f"{base_name}_analysis.log",
        convergence=convergence
    )
    atoms.calc = calc
    
    parprint("Running SCF calculation...")
    E_total = atoms.get_potential_energy()
    parprint(f"Total energy: {E_total:.6f} eV")
    
    # Calculate DOS
    parprint("Calculating DOS...")
    
    fermi = calc.get_fermi_level()
    
    # Collect all eigenvalues from all k-points and spins
    parprint("  Extracting eigenvalues from all k-points...")
    eigenvalues = []
    kpt_weights = []
    
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
                    kpt_weights.append(w)
            except Exception as e:
                parprint(f"  Warning: Could not get eigenvalues for k={k}, s={s}: {e}")
    
    eigenvalues = np.array(eigenvalues)
    kpt_weights = np.array(kpt_weights)
    
    parprint(f"  Found {len(eigenvalues)} eigenvalues from {n_kpts} k-points")
    parprint(f"  Eigenvalue range: [{eigenvalues.min():.2f}, {eigenvalues.max():.2f}] eV")
    
    # Create energy grid
    e_min = eigenvalues.min() - 2.0
    e_max = eigenvalues.max() + 2.0
    npts = 1001
    energies = np.linspace(e_min, e_max, npts)
    
    # Compute DOS with Gaussian broadening (weighted by k-point weights)
    dos = np.zeros_like(energies)
    for eig, w in zip(eigenvalues, kpt_weights):
        dos += w * np.exp(-((energies - eig) / dos_width)**2 / 2)
    dos /= (dos_width * np.sqrt(2 * np.pi))
    
    # Find band edges from eigenvalues directly (more reliable)
    # VBM: highest occupied state (below or at Fermi level)
    # CBM: lowest unoccupied state (above Fermi level)
    occupied = eigenvalues[eigenvalues <= fermi + 0.01]  # Small tolerance
    unoccupied = eigenvalues[eigenvalues > fermi + 0.01]
    
    if len(occupied) > 0:
        vbm = occupied.max()
    else:
        vbm = fermi
    
    if len(unoccupied) > 0:
        cbm = unoccupied.min()
    else:
        cbm = fermi
    
    band_gap = cbm - vbm if cbm > vbm else 0.0
    
    parprint(f"  VBM: {vbm:.4f} eV, CBM: {cbm:.4f} eV, Gap: {band_gap:.4f} eV")
    
    parprint(f"\nElectronic structure:")
    parprint(f"  Fermi level: {fermi:.4f} eV")
    parprint(f"  VBM:         {vbm:.4f} eV")
    parprint(f"  CBM:         {cbm:.4f} eV")
    parprint(f"  Band gap:    {band_gap:.4f} eV")
    
    summary['energies'] = {'total_energy_eV': float(E_total), 'fermi_level_eV': float(fermi)}
    summary['electronic_structure'] = {
        'band_gap_eV': float(band_gap),
        'vbm_eV': float(vbm),
        'cbm_eV': float(cbm),
        'fermi_level_eV': float(fermi)
    }
    
    # Vacuum alignment
    parprint("\nCalculating vacuum level...")
    try:
        v = calc.get_electrostatic_potential()
        if v is not None:
            v_z = v.mean(axis=(0, 1))
            V_vacuum = v_z.max()
            
            VBM_vacuum = vbm - V_vacuum
            CBM_vacuum = cbm - V_vacuum
            work_function = V_vacuum - fermi
            
            H_redox = -4.44  # H+/H2 vs vacuum
            O_redox = -5.67  # O2/H2O vs vacuum
            
            parprint(f"  Vacuum level:    {V_vacuum:.4f} eV")
            parprint(f"  VBM vs vacuum:   {VBM_vacuum:.4f} eV")
            parprint(f"  CBM vs vacuum:   {CBM_vacuum:.4f} eV")
            parprint(f"  Work function:   {work_function:.4f} eV")
            
            summary['vacuum_alignment'] = {
                'vacuum_level_eV': float(V_vacuum),
                'vbm_vs_vacuum_eV': float(VBM_vacuum),
                'cbm_vs_vacuum_eV': float(CBM_vacuum),
                'work_function_eV': float(work_function),
                'can_drive_HER': bool(CBM_vacuum > H_redox),
                'can_drive_OER': bool(VBM_vacuum < O_redox)
            }
    except Exception as e:
        parprint(f"  Warning: Could not calculate vacuum level: {e}")
    
    # Save DOS plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.fill_between(energies, dos, alpha=0.3, color='steelblue')
        ax.plot(energies, dos, color='steelblue', linewidth=1)
        ax.axvline(fermi, color='red', linestyle='--', label=f'Fermi ({fermi:.2f} eV)')
        ax.axvline(vbm, color='green', linestyle=':', label=f'VBM ({vbm:.2f} eV)')
        ax.axvline(cbm, color='orange', linestyle=':', label=f'CBM ({cbm:.2f} eV)')
        if band_gap > 0.1:
            ax.axvspan(vbm, cbm, alpha=0.15, color='yellow', label=f'Gap = {band_gap:.2f} eV')
        ax.set_xlabel('Energy (eV)')
        ax.set_ylabel('DOS (states/eV)')
        ax.set_title(f"DOS: {atoms.get_chemical_formula()} (E_g = {band_gap:.2f} eV)")
        ax.legend()
        ax.set_xlim(fermi - 8, fermi + 8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{base_name}_dos.png", dpi=150)
        plt.close()
        parprint(f"\nSaved: {base_name}_dos.png")
    except Exception as e:
        parprint(f"Warning: Could not create DOS plot: {e}")
    
    # =========================================================
    # Step 2: Formation Energy (if pristine provided)
    # =========================================================
    if args.pristine:
        parprint(f"\n{'='*70}")
        parprint("STEP 2: Formation Energy")
        parprint(f"{'='*70}")
        
        parprint(f"Calculating pristine energy: {args.pristine}")
        pristine_atoms = io.read(args.pristine)
        
        pristine_calc = GPAW(
            mode=PW(ecut),
            xc='PBE',
            occupations=FermiDirac(smearing),
            kpts=kpts_dos,
            txt=f"{base_name}_pristine.log",
            convergence=convergence
        )
        pristine_atoms.calc = pristine_calc
        E_pristine = pristine_atoms.get_potential_energy()
        
        # N-rich conditions
        mu_N = -8.3  # eV (from N2)
        n_removed = pristine_atoms.get_chemical_symbols().count('N') - atoms.get_chemical_symbols().count('N')
        
        E_formation = E_total - E_pristine + n_removed * mu_N
        
        parprint(f"  E(defect):    {E_total:.6f} eV")
        parprint(f"  E(pristine):  {E_pristine:.6f} eV")
        parprint(f"  N removed:    {n_removed}")
        parprint(f"  μ_N:          {mu_N:.2f} eV (N-rich)")
        parprint(f"  E_formation:  {E_formation:.4f} eV")
        
        summary['formation_energy'] = {
            'E_formation_eV': float(E_formation),
            'E_defect_eV': float(E_total),
            'E_pristine_eV': float(E_pristine),
            'mu_N_eV': float(mu_N),
            'n_removed': n_removed
        }
    
    # =========================================================
    # Step 3: H Adsorption (if site specified)
    # =========================================================
    if args.h_site is not None and not args.dos_only:
        parprint(f"\n{'='*70}")
        parprint("STEP 3: H Adsorption (ΔG_H*)")
        parprint(f"{'='*70}")
        
        from ase import Atom
        from ase.optimize import BFGS
        from ase.constraints import FixAtoms
        
        # Add H above specified site
        h_pos = atoms.positions[args.h_site] + np.array([0, 0, 1.1])
        parprint(f"Adding H above atom {args.h_site}")
        parprint(f"H position: ({h_pos[0]:.3f}, {h_pos[1]:.3f}, {h_pos[2]:.3f})")
        parprint(f"H optimization: fmax={h_opt_fmax}, max_steps={h_opt_steps}")
        
        atoms_H = atoms.copy()
        atoms_H.append(Atom('H', h_pos))
        
        # Calculate energy with H
        calc_H = GPAW(
            mode=PW(ecut),
            xc='PBE',
            occupations=FermiDirac(smearing),
            kpts=kpts_scf,
            txt=f"{base_name}_with_H.log",
            convergence=convergence
        )
        atoms_H.calc = calc_H
        
        # Optimize H position
        constraint = FixAtoms(indices=list(range(len(atoms_H) - 1)))
        atoms_H.set_constraint(constraint)
        
        opt = BFGS(atoms_H, logfile=None)
        opt.run(fmax=h_opt_fmax, steps=h_opt_steps)
        
        E_surf_H = atoms_H.get_potential_energy()
        
        # Calculate ΔG_H*
        E_H2 = -6.77  # PBE reference
        delta_E_H = E_surf_H - E_total - 0.5 * E_H2
        delta_G_H = delta_E_H + 0.24  # ZPE + entropy correction
        
        parprint(f"\n  E(surface):   {E_total:.6f} eV")
        parprint(f"  E(surface+H): {E_surf_H:.6f} eV")
        parprint(f"  ΔE_H:         {delta_E_H:.4f} eV")
        parprint(f"  ΔG_H*:        {delta_G_H:.4f} eV")
        parprint(f"  |ΔG_H*|:      {abs(delta_G_H):.4f} eV")
        
        if abs(delta_G_H) < 0.1:
            parprint(f"  >>> EXCELLENT HER activity!")
        elif abs(delta_G_H) < 0.2:
            parprint(f"  >>> GOOD HER activity")
        
        summary['her_activity'] = {
            'delta_E_H_eV': float(delta_E_H),
            'delta_G_H_eV': float(delta_G_H),
            'h_site_index': args.h_site,
            'E_surface_H_eV': float(E_surf_H)
        }
        
        io.write(f"{base_name}_with_H_opt.cif", atoms_H)
    
    # =========================================================
    # Step 4: Charge Analysis
    # =========================================================
    if not args.skip_charges:
        parprint(f"\n{'='*70}")
        parprint("STEP 4: Charge Analysis (Hirshfeld)")
        parprint(f"{'='*70}")
        
        try:
            from gpaw.analyse.hirshfeld import HirshfeldPartitioning
            
            hp = HirshfeldPartitioning(calc)
            charges = hp.get_charges()
            
            symbols = atoms.get_chemical_symbols()
            
            # Group by element
            charge_by_element = {}
            for sym, q in zip(symbols, charges):
                if sym not in charge_by_element:
                    charge_by_element[sym] = []
                charge_by_element[sym].append(q)
            
            parprint("\nCharges by element:")
            analysis = {}
            for sym, qs in charge_by_element.items():
                qs = np.array(qs)
                analysis[sym] = {
                    'count': len(qs),
                    'mean': float(qs.mean()),
                    'std': float(qs.std()),
                    'min': float(qs.min()),
                    'max': float(qs.max())
                }
                parprint(f"  {sym}: mean={qs.mean():.4f}e, std={qs.std():.4f}e")
            
            summary['charge_analysis'] = {
                'method': 'hirshfeld',
                'by_element': analysis,
                'charges': charges.tolist()
            }
            
        except Exception as e:
            parprint(f"Warning: Charge analysis failed: {e}")
    
    # =========================================================
    # Save results
    # =========================================================
    parprint(f"\n{'='*70}")
    parprint("SAVING RESULTS")
    parprint(f"{'='*70}")
    
    # Save summary JSON
    summary_file = f"{base_name}_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    parprint(f"Saved: {summary_file}")
    
    # Generate report
    report_file = f"{base_name}_report.txt"
    generate_report(summary, report_file)
    
    parprint(f"\n{'='*70}")
    parprint("ANALYSIS COMPLETE")
    parprint(f"{'='*70}")
    
    if quality_warning:
        parprint(f"\n{quality_warning}")
    
    parprint(f"\nQuality: {args.quality} ({preset['uncertainty']})")
    parprint(f"\nOutput files:")
    parprint(f"  - {summary_file}")
    parprint(f"  - {report_file}")
    parprint(f"  - {base_name}_dos.png")
    
    if args.quality == 'draft':
        parprint(f"\n>>> Re-run with --quality standard for publication values")
    
    return summary


if __name__ == "__main__":
    main()

