#!/usr/bin/env python3
"""
Calculate atomic charges for defect analysis.

Methods available:
    1. Bader charge (via GPAW cube file export + external bader tool)
    2. Hirshfeld charges (built into GPAW)
    3. Mulliken charges (built into GPAW, LCAO mode only)

For true Bader analysis:
    - Export electron density to cube file
    - Run external 'bader' program (Henkelman group)
    - Parse ACF.dat output

Usage:
    # Fast draft calculation
    gpaw python calculate_charges.py optimized.cif --quality draft
    
    # Hirshfeld charges (fast, no external tools)
    gpaw python calculate_charges.py optimized.cif --method hirshfeld
    
    # Export cube file for Bader analysis
    gpaw python calculate_charges.py optimized.cif --export-cube
    
    # Then run: bader CHGCAR.cube (external tool)
    # Parse results: python calculate_charges.py --parse-bader ACF.dat

Quality presets affect SCF convergence (charges are less sensitive than energies).

Reference:
    - Bader: Henkelman et al., Comput. Mater. Sci. 36, 354 (2006)
    - Hirshfeld: Hirshfeld, Theor. Chim. Acta 44, 129 (1977)
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

from gpaw import GPAW, PW, FermiDirac

from calc_presets import (
    get_preset, detect_system_type, print_preset_info,
    get_quality_warning, QUALITY_CHOICES
)


def calculate_hirshfeld_charges(atoms, calc):
    """
    Calculate Hirshfeld charges.
    
    Hirshfeld partitioning divides electron density based on 
    pro-molecular reference densities.
    
    Returns array of charges for each atom.
    """
    from gpaw.analyse.hirshfeld import HirshfeldPartitioning
    
    parprint("Calculating Hirshfeld charges...")
    
    # Hirshfeld partitioning
    hp = HirshfeldPartitioning(calc)
    charges = hp.get_charges()
    
    return charges


def export_cube_for_bader(atoms, calc, filename='density.cube'):
    """
    Export electron density to cube file for external Bader analysis.
    
    Use with Henkelman group's bader program:
        bader density.cube
    
    This creates ACF.dat with Bader charges.
    """
    from gpaw.utilities.ps2ae import PS2AE
    
    parprint(f"Exporting electron density to {filename}...")
    
    # Get all-electron density (more accurate for Bader)
    try:
        # Try to get all-electron density
        ps2ae = PS2AE(calc)
        density = ps2ae.get_ae_density()
        parprint("Using all-electron density")
    except:
        # Fall back to pseudo-density
        density = calc.get_pseudo_density()
        parprint("Using pseudo-density (all-electron not available)")
    
    # Write to cube file
    from ase.io.cube import write_cube
    with open(filename, 'w') as f:
        write_cube(f, atoms, data=density)
    
    parprint(f"Saved: {filename}")
    parprint(f"\n>>> For Bader analysis:")
    parprint(f"  1. Install bader: http://theory.cm.utexas.edu/henkelman/code/bader/")
    parprint(f"  2. Run: bader {filename}")
    parprint(f"  3. Results in: ACF.dat")
    
    return filename


def parse_bader_acf(acf_file, atoms):
    """
    Parse Bader analysis output (ACF.dat from Henkelman's bader).
    
    ACF.dat format:
        #    X         Y         Z       CHARGE      MIN DIST   ATOMIC VOL
        1   0.0000    0.0000    0.0000    6.0000      1.2345     12.345
        ...
    """
    parprint(f"Parsing Bader results from {acf_file}...")
    
    charges = []
    volumes = []
    
    with open(acf_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if line.startswith('#') or line.startswith('-') or not line:
            continue
        if 'VACUUM' in line or 'NUMBER' in line:
            continue
        
        parts = line.split()
        if len(parts) >= 5:
            try:
                idx = int(parts[0])
                charge = float(parts[4])  # CHARGE column
                charges.append(charge)
                if len(parts) >= 7:
                    volumes.append(float(parts[6]))
            except (ValueError, IndexError):
                continue
    
    charges = np.array(charges)
    
    # Convert to net charge (electrons - valence)
    symbols = atoms.get_chemical_symbols()
    valence = {'H': 1, 'C': 4, 'N': 5, 'O': 6, 'S': 6, 'P': 5, 
               'B': 3, 'Fe': 8, 'Co': 9, 'Ni': 10, 'Cu': 11, 'Pt': 10}
    
    net_charges = []
    for i, sym in enumerate(symbols):
        if i < len(charges):
            v = valence.get(sym, 0)
            net_charges.append(charges[i] - v)
    
    return np.array(net_charges), charges


def analyze_defect_charges(atoms, charges, defect_indices=None):
    """
    Analyze charge distribution around defect.
    
    Args:
        atoms: ASE Atoms object
        charges: array of atomic charges
        defect_indices: list of atom indices near defect (optional)
    
    Returns dict with charge analysis.
    """
    symbols = atoms.get_chemical_symbols()
    
    # Group by element
    charge_by_element = {}
    for sym, q in zip(symbols, charges):
        if sym not in charge_by_element:
            charge_by_element[sym] = []
        charge_by_element[sym].append(q)
    
    analysis = {
        'total_charge': float(charges.sum()),
        'by_element': {}
    }
    
    for sym, qs in charge_by_element.items():
        qs = np.array(qs)
        analysis['by_element'][sym] = {
            'count': len(qs),
            'mean': float(qs.mean()),
            'std': float(qs.std()),
            'min': float(qs.min()),
            'max': float(qs.max()),
            'total': float(qs.sum())
        }
    
    # Identify unusual charges (potential defect signatures)
    for sym in set(symbols):
        qs = np.array([charges[i] for i, s in enumerate(symbols) if s == sym])
        mean_q = qs.mean()
        std_q = qs.std()
        
        # Find outliers (> 2 sigma from mean)
        outliers = []
        for i, s in enumerate(symbols):
            if s == sym and abs(charges[i] - mean_q) > 2 * std_q:
                outliers.append({
                    'index': i,
                    'charge': float(charges[i]),
                    'deviation': float(charges[i] - mean_q)
                })
        
        if outliers:
            analysis['by_element'][sym]['outliers'] = outliers
    
    return analysis


def print_charge_table(atoms, charges, title="Atomic Charges"):
    """Print formatted table of charges."""
    symbols = atoms.get_chemical_symbols()
    positions = atoms.positions
    
    parprint(f"\n{'='*70}")
    parprint(title)
    parprint(f"{'='*70}")
    parprint(f"{'Index':>6} {'Symbol':>6} {'Charge':>10} {'X':>8} {'Y':>8} {'Z':>8}")
    parprint("-"*70)
    
    for i, (sym, q, pos) in enumerate(zip(symbols, charges, positions)):
        parprint(f"{i:>6} {sym:>6} {q:>10.4f} {pos[0]:>8.3f} {pos[1]:>8.3f} {pos[2]:>8.3f}")
    
    parprint("-"*70)
    parprint(f"Total charge: {charges.sum():.4f} e")
    

def main():
    parser = argparse.ArgumentParser(
        description="Calculate atomic charges for defect analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("cif_file", nargs='?', help="Input structure (CIF/XYZ)")
    parser.add_argument("--quality", choices=QUALITY_CHOICES, default='standard',
                        help="Quality preset: draft (fast), standard, production (default: standard)")
    parser.add_argument("--method", choices=['hirshfeld', 'bader'], default='hirshfeld',
                        help="Charge analysis method (default: hirshfeld)")
    parser.add_argument("--export-cube", action="store_true",
                        help="Export electron density cube for external Bader")
    parser.add_argument("--parse-bader", type=str, metavar='ACF.dat',
                        help="Parse Bader ACF.dat output file")
    parser.add_argument("--reference", type=str, default=None,
                        help="Reference structure for charge comparison")
    parser.add_argument("--ecut", type=float, default=None,
                        help="Plane-wave cutoff in eV (overrides preset)")
    parser.add_argument("--kpts", type=int, nargs=3, default=None,
                        help="k-point grid (overrides preset)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print info only, don't run calculation")
    
    args = parser.parse_args()
    
    # Parse existing Bader output
    if args.parse_bader:
        if not args.cif_file:
            parprint("ERROR: Need structure file to interpret Bader results")
            return
        
        atoms = io.read(args.cif_file)
        charges, raw_charges = parse_bader_acf(args.parse_bader, atoms)
        
        print_charge_table(atoms, charges, "Bader Net Charges")
        
        # Analyze
        analysis = analyze_defect_charges(atoms, charges)
        
        parprint(f"\n{'='*50}")
        parprint("CHARGE ANALYSIS BY ELEMENT")
        parprint(f"{'='*50}")
        for sym, data in analysis['by_element'].items():
            parprint(f"\n  {sym}:")
            parprint(f"    Count: {data['count']}")
            parprint(f"    Mean:  {data['mean']:.4f} e")
            parprint(f"    Std:   {data['std']:.4f} e")
            parprint(f"    Range: [{data['min']:.4f}, {data['max']:.4f}] e")
            if 'outliers' in data:
                parprint(f"    Outliers (>2σ):")
                for o in data['outliers']:
                    parprint(f"      - Index {o['index']}: {o['charge']:.4f} e (Δ={o['deviation']:+.4f})")
        
        # Save
        base_name = os.path.splitext(os.path.basename(args.cif_file))[0]
        results = {
            'method': 'bader',
            'charges': charges.tolist(),
            'raw_bader_charges': raw_charges.tolist(),
            'analysis': analysis
        }
        with open(f"{base_name}_bader_charges.json", 'w') as f:
            json.dump(results, f, indent=2)
        parprint(f"\nSaved: {base_name}_bader_charges.json")
        return
    
    if not args.cif_file:
        parser.print_help()
        return
    
    # Read structure
    parprint(f"Reading structure: {args.cif_file}")
    atoms = io.read(args.cif_file)
    base_name = os.path.splitext(os.path.basename(args.cif_file))[0]
    
    # Detect system type and get preset
    system_type = detect_system_type(atoms)
    preset = get_preset(args.quality, system_type)
    
    # Override with command-line arguments if provided
    ecut = args.ecut if args.ecut is not None else preset['ecut']
    kpts = tuple(args.kpts) if args.kpts is not None else preset['kpts']
    convergence = preset['convergence']
    smearing = preset['smearing']
    
    parprint(f"\nStructure: {atoms.get_chemical_formula()}")
    parprint(f"Atoms: {len(atoms)}")
    parprint(f"System type: {system_type}")
    
    # Print preset info
    print_preset_info(preset, args.quality)
    
    parprint(f"\nEffective parameters:")
    parprint(f"  ecut: {ecut} eV")
    parprint(f"  kpts: {kpts}")
    
    if args.dry_run:
        parprint("\n[DRY RUN] Exiting.")
        return
    
    # =========================================================
    # Run SCF calculation
    # =========================================================
    parprint("\nRunning SCF calculation...")
    
    calc = GPAW(
        mode=PW(ecut),
        xc='PBE',
        occupations=FermiDirac(smearing),
        kpts=kpts,
        txt=f"{base_name}_charges.log",
        convergence=convergence
    )
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    parprint(f"Total energy: {energy:.6f} eV")
    
    # =========================================================
    # Export cube file for Bader
    # =========================================================
    if args.export_cube:
        cube_file = f"{base_name}_density.cube"
        export_cube_for_bader(atoms, calc, cube_file)
        return
    
    # =========================================================
    # Calculate charges
    # =========================================================
    if args.method == 'hirshfeld':
        charges = calculate_hirshfeld_charges(atoms, calc)
        method_name = "Hirshfeld"
    else:
        parprint("For Bader charges, use --export-cube then external bader tool")
        parprint("Or use --parse-bader ACF.dat to process existing results")
        return
    
    # Print results
    print_charge_table(atoms, charges, f"{method_name} Charges")
    
    # Analyze
    analysis = analyze_defect_charges(atoms, charges)
    
    parprint(f"\n{'='*50}")
    parprint("CHARGE ANALYSIS BY ELEMENT")
    parprint(f"{'='*50}")
    for sym, data in analysis['by_element'].items():
        parprint(f"\n  {sym}:")
        parprint(f"    Count: {data['count']}")
        parprint(f"    Mean:  {data['mean']:.4f} e")
        parprint(f"    Std:   {data['std']:.4f} e")
        parprint(f"    Range: [{data['min']:.4f}, {data['max']:.4f}] e")
        if 'outliers' in data:
            parprint(f"    Outliers (>2σ):")
            for o in data['outliers']:
                parprint(f"      - Index {o['index']}: {o['charge']:.4f} e (Δ={o['deviation']:+.4f})")
    
    # Compare with reference if provided
    if args.reference:
        parprint(f"\nComparing with reference: {args.reference}")
        ref_atoms = io.read(args.reference)
        
        ref_calc = GPAW(
            mode=PW(ecut),
            xc='PBE',
            occupations=FermiDirac(smearing),
            kpts=kpts,
            txt=f"{base_name}_ref_charges.log",
            convergence=convergence
        )
        ref_atoms.calc = ref_calc
        ref_atoms.get_potential_energy()
        
        ref_charges = calculate_hirshfeld_charges(ref_atoms, ref_calc)
        
        parprint(f"\n{'='*50}")
        parprint("CHARGE TRANSFER (vs reference)")
        parprint(f"{'='*50}")
        
        # This only works if structures have same atoms
        if len(charges) == len(ref_charges):
            delta_q = charges - ref_charges
            for i, (sym, dq) in enumerate(zip(atoms.get_chemical_symbols(), delta_q)):
                if abs(dq) > 0.05:  # Significant change
                    parprint(f"  Atom {i} ({sym}): Δq = {dq:+.4f} e")
    
    # =========================================================
    # Save results
    # =========================================================
    quality_warning = get_quality_warning(args.quality)
    
    results = {
        'quality': args.quality,
        'quality_warning': quality_warning,
        'structure': args.cif_file,
        'method': args.method,
        'charges': charges.tolist(),
        'analysis': analysis,
        'calculation': {
            'ecut': ecut,
            'kpts': list(kpts)
        }
    }
    
    results_file = f"{base_name}_{args.method}_charges.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    parprint(f"\nSaved: {results_file}")
    
    if quality_warning:
        parprint(f"\n{quality_warning}")
    
    # =========================================================
    # Summary for defect analysis
    # =========================================================
    parprint(f"\n{'='*50}")
    parprint("DEFECT CHARGE SIGNATURES")
    parprint(f"{'='*50}")
    
    symbols = atoms.get_chemical_symbols()
    for sym, data in analysis['by_element'].items():
        if 'outliers' in data:
            parprint(f"\n  {sym} atoms with unusual charges (possible defect sites):")
            for o in data['outliers']:
                idx = o['index']
                pos = atoms.positions[idx]
                parprint(f"    Index {idx}: q={o['charge']:.3f}e at ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
    
    parprint("\nDone.")


if __name__ == "__main__":
    main()

