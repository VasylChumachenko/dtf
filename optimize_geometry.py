"""
Geometry optimization script for CIF file using GPAW.

Usage:
    gpaw python optimize_geometry.py input.cif
    mpirun -n 4 gpaw python -- optimize_geometry.py input.cif [options]

Options:
    --quality    Quality preset: draft (fast), standard, production (default: standard)
    --ecut       Plane-wave cutoff in eV (overrides preset)
    --kpts       k-point grid, e.g. 4 4 4 (overrides preset)
    --smearing   Fermi-Dirac width in eV (default: 0.05)
    --fmax       Force convergence in eV/Å (overrides preset)
    --steps      Max optimization steps (default: 200)
    --fix-c      Fix c parameter (for 2D/slab with vacuum along c)
    --fix-cell   Fix ALL cell parameters (for defect calculations)
    --lcao       Use LCAO mode instead of plane-wave (faster, GPU-capable)
    --basis      LCAO basis set: sz, dzp, tzp (default: dzp)
    --gpu        Enable GPU acceleration (LCAO only, requires CuPy)
    --two-stage  First relax positions, then cell+positions
    --fire-only  Use FIRE only (robust for defects, H atoms, oscillating systems)
    --supercell  Create supercell before optimization (e.g., 2 2 1)
    --dry-run    Print structure info only, don't run calculation
    --symmetry   Use symmetry to reduce k-points (faster, but not for cell optimization)

Output (in {name}/ folder):
    - {name}_{quality}_optimized.cif  : Optimized structure
    - {name}_{quality}_optimized.xyz  : For visualization
    - {name}_{quality}_gpaw.log       : GPAW log

Example:
    # Fast draft optimization
    mpirun -n 4 gpaw python -- optimize_geometry.py g_c3n4.cif --quality draft --fix-c

    # Standard optimization
    mpirun -n 4 gpaw python -- optimize_geometry.py C3N4.cif --quality standard
    
    # 2D slab with supercell
    mpirun -n 4 gpaw python -- optimize_geometry.py slab.cif --fix-c --kpts 6 6 1 --supercell 2 2 1
"""

import argparse
import os
import sys
import time
import numpy as np

from ase import io
from ase.optimize import LBFGS, FIRE
from ase.constraints import ExpCellFilter  # More stable than UnitCellFilter
from ase.parallel import parprint, world

from gpaw import GPAW, PW, FermiDirac

# Quality presets for optimization
QUALITY_PRESETS = {
    'draft': {
        'ecut': 300,
        'kpts_2d': (2, 2, 1),
        'kpts_3d': (2, 2, 2),
        'fmax': 0.1,
        'convergence': {'energy': 1e-3, 'density': 1e-3},
        'use_fast': True,
        'description': 'Fast screening (~5× faster, ±0.15 eV)'
    },
    'standard': {
        'ecut': 500,
        'kpts_2d': (4, 4, 1),
        'kpts_3d': (4, 4, 4),
        'fmax': 0.03,
        'convergence': {'energy': 1e-4, 'density': 1e-4},
        'use_fast': False,
        'description': 'Standard quality (default)'
    },
    'production': {
        'ecut': 600,
        'kpts_2d': (6, 6, 1),
        'kpts_3d': (6, 6, 6),
        'fmax': 0.01,
        'convergence': {'energy': 1e-5, 'density': 1e-5},
        'use_fast': False,
        'description': 'Publication quality'
    }
}

# Try to import tqdm for progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


class ProgressCallback:
    """Progress tracker for ASE optimizers with optional tqdm bar."""
    
    def __init__(self, atoms, max_steps, fmax_target, label="Optimizing", use_bar=True):
        self.atoms = atoms
        self.max_steps = max_steps
        self.fmax_target = fmax_target
        self.label = label
        self.step = 0
        self.start_time = time.time()
        self.use_bar = use_bar and HAS_TQDM and world.rank == 0
        
        if self.use_bar:
            self.pbar = tqdm(
                total=max_steps,
                desc=label,
                unit="step",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
                file=sys.stdout
            )
        else:
            self.pbar = None
    
    def __call__(self):
        """Called after each optimization step."""
        self.step += 1
        
        # Get current forces
        forces = self.atoms.get_forces()
        fmax = np.sqrt((forces ** 2).sum(axis=1)).max()
        energy = self.atoms.get_potential_energy()
        
        elapsed = time.time() - self.start_time
        
        if self.use_bar and self.pbar is not None:
            self.pbar.update(1)
            self.pbar.set_postfix({
                'fmax': f'{fmax:.4f}',
                'E': f'{energy:.3f}',
                'target': f'{self.fmax_target}'
            })
        elif world.rank == 0:
            # Simple text progress for non-tqdm
            pct = 100 * self.step / self.max_steps
            bar_len = 30
            filled = int(bar_len * self.step / self.max_steps)
            bar = '█' * filled + '░' * (bar_len - filled)
            print(f"\r{self.label}: [{bar}] {self.step}/{self.max_steps} "
                  f"fmax={fmax:.4f} E={energy:.3f} eV ({elapsed:.0f}s)", 
                  end='', flush=True)
    
    def close(self):
        """Clean up progress bar."""
        if self.use_bar and self.pbar is not None:
            self.pbar.close()
        elif world.rank == 0:
            print()  # New line after text progress


def get_min_distance(atoms):
    """Get minimum interatomic distance (catches bad geometries)."""
    dmin = float('inf')
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            d = atoms.get_distance(i, j, mic=True)
            dmin = min(dmin, d)
    return dmin


def pick_kpt_parallel(world_size: int, kpts) -> dict:
    """
    Choose safe k-point parallelization.
    GPAW requires ALL CPUs to be used:
    world_size = kpt_groups * domain_groups * band_groups
    """
    nk = int(np.prod(kpts))
    if world_size <= 1:
        return None

    # Find largest kpt_groups that divides BOTH nk and world_size
    valid_kpt_groups = [d for d in range(1, min(nk, world_size) + 1) 
                        if nk % d == 0 and world_size % d == 0]
    kpt_groups = max(valid_kpt_groups) if valid_kpt_groups else 1

    # Remaining processes go to domain decomposition
    domain_groups = world_size // kpt_groups
    
    # Sanity check: kpt * domain must equal world_size
    if kpt_groups * domain_groups != world_size:
        # Can't parallelize properly, let GPAW decide
        return None

    return {'kpt': kpt_groups, 'domain': domain_groups, 'band': 1}


def main():
    parser = argparse.ArgumentParser(
        description="Geometry optimization using GPAW",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("cif_file", nargs="?", default="C3N4_mp-1193580_primitive.cif.txt",
                        help="Input CIF file")
    parser.add_argument("--quality", choices=['draft', 'standard', 'production'], default='standard',
                        help="Quality preset (default: standard)")
    parser.add_argument("--ecut", type=float, default=None,
                        help="Plane-wave cutoff in eV (overrides preset)")
    parser.add_argument("--kpts", type=int, nargs=3, default=None,
                        help="k-point grid (overrides preset)")
    parser.add_argument("--smearing", type=float, default=0.05,
                        help="Fermi-Dirac width in eV (default: 0.05)")
    parser.add_argument("--fmax", type=float, default=None,
                        help="Force convergence in eV/Å (overrides preset)")
    parser.add_argument("--steps", type=int, default=200,
                        help="Max optimization steps (default: 200)")
    parser.add_argument("--fix-c", action="store_true",
                        help="Fix c parameter (for 2D/slab with vacuum along c)")
    parser.add_argument("--fix-cell", action="store_true",
                        help="Fix ALL cell parameters (for defect calculations in supercells)")
    parser.add_argument("--lcao", action="store_true",
                        help="Use LCAO mode instead of plane-wave (5-10× faster)")
    parser.add_argument("--basis", type=str, default="dzp",
                        choices=['sz', 'szp', 'dz', 'dzp', 'tzp', 'qzp'],
                        help="LCAO basis set (default: dzp)")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU acceleration (LCAO mode only, requires CuPy)")
    parser.add_argument("--two-stage", action="store_true",
                        help="Two-stage: first positions only, then cell+positions")
    parser.add_argument("--fire-only", action="store_true",
                        help="Use only FIRE optimizer (more robust for defects/H atoms)")
    parser.add_argument("--supercell", type=int, nargs=3, default=None,
                        metavar=("NX", "NY", "NZ"),
                        help="Create supercell before optimization (e.g., 2 2 1)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print structure info only, don't run calculation")
    parser.add_argument("--symmetry", action="store_true",
                        help="Use symmetry to reduce k-points (faster, but incompatible with cell optimization)")
    args = parser.parse_args()

    cif_file = args.cif_file
    
    # Get quality preset
    preset = QUALITY_PRESETS[args.quality]
    parprint(f"\n{'='*60}")
    parprint(f"GEOMETRY OPTIMIZATION - {args.quality.upper()} mode")
    parprint(f"{'='*60}")
    parprint(f"Preset: {preset['description']}")

    # Generate base name and output folder
    base_name = os.path.splitext(os.path.basename(cif_file))[0]
    if base_name.endswith(".cif"):
        base_name = base_name[:-4]
    
    # Create output directory (don't delete if exists)
    output_dir = base_name
    if world.rank == 0:
        os.makedirs(output_dir, exist_ok=True)
    world.barrier()
    
    # Quality suffix for output files
    quality_suffix = f"_{args.quality}"
    output_base = os.path.join(output_dir, f"{base_name}{quality_suffix}")
    
    parprint(f"Output directory: {output_dir}/")
    
    # Read structure first to detect 2D vs 3D
    parprint(f"\nReading structure from {cif_file}...")
    atoms = io.read(cif_file, format="cif")
    
    # Detect if 2D (large vacuum in z)
    # For 2D slabs: c should be significantly larger than typical bond lengths
    # AND there should be a gap in z-coordinates (vacuum)
    cell_lengths = atoms.cell.lengths()
    z_coords = atoms.positions[:, 2]
    z_range = z_coords.max() - z_coords.min() if len(atoms) > 0 else 0
    # 2D if: c > 12 Å AND (c much larger than z-extent of atoms, indicating vacuum)
    is_2d = cell_lengths[2] > 12.0 and (cell_lengths[2] - z_range) > 8.0
    
    # Apply preset values (can be overridden by command line)
    if args.ecut is not None:
        ecut = args.ecut
    else:
        ecut = preset['ecut']
    
    if args.kpts is not None:
        kpts = tuple(args.kpts)
    else:
        kpts = preset['kpts_2d'] if is_2d else preset['kpts_3d']
    
    if args.fmax is not None:
        fmax = args.fmax
    else:
        fmax = preset['fmax']
    
    convergence = preset['convergence']
    use_fast = preset['use_fast']
    
    # Output filenames
    output_cif = f"{output_base}_optimized.cif"
    output_xyz = f"{output_base}_optimized.xyz"
    gpaw_log = f"{output_base}_gpaw.log"
    traj_file = f"{output_base}_optimization.traj"

    # Create supercell if requested
    if args.supercell is not None:
        sc = tuple(args.supercell)
        n_orig = len(atoms)
        atoms = atoms * sc
        parprint(f"Created {sc[0]}×{sc[1]}×{sc[2]} supercell: {n_orig} → {len(atoms)} atoms")
        # Update output names to include supercell info
        supercell_str = f"_{sc[0]}x{sc[1]}x{sc[2]}"
        output_cif = f"{output_base}{supercell_str}_optimized.cif"
        output_xyz = f"{output_base}{supercell_str}_optimized.xyz"
        gpaw_log = f"{output_base}{supercell_str}_gpaw.log"
        traj_file = f"{output_base}{supercell_str}_optimization.traj"

    # Check for bad geometry (too short distances)
    dmin = get_min_distance(atoms)
    parprint(f"\nGeometry check:")
    parprint(f"  Min interatomic distance: {dmin:.3f} Å")
    if dmin < 0.8:
        parprint(f"  WARNING: Very short distance detected! May cause convergence issues.")
    elif dmin < 1.0:
        parprint(f"  WARNING: Short distance detected. Consider pre-relaxation.")

    # Print initial structure
    parprint(f"\nInitial structure:")
    parprint(f"  Number of atoms: {len(atoms)}")
    parprint(f"  Chemical formula: {atoms.get_chemical_formula()}")
    parprint(f"  Cell (a, b, c): {atoms.cell.lengths()}")
    parprint(f"  Angles (α, β, γ): {atoms.cell.angles()}")
    parprint(f"  Volume: {atoms.get_volume():.3f} Å³")

    # Print calculation parameters
    parprint(f"\nCalculation parameters:")
    parprint(f"  Quality: {args.quality}")
    parprint(f"  System: {'2D slab' if is_2d else '3D bulk'}")
    if args.lcao:
        gpu_info = " + GPU" if args.gpu else ""
        parprint(f"  Mode: LCAO (basis={args.basis}){gpu_info}")
    else:
        parprint(f"  Mode: PW (ecut={ecut} eV)")
        if args.gpu:
            parprint(f"  WARNING: --gpu only works with --lcao mode")
    parprint(f"  kpts: {kpts} ({np.prod(kpts)} k-points)")
    parprint(f"  smearing: {args.smearing} eV")
    parprint(f"  fmax: {fmax} eV/Å")
    parprint(f"  convergence: {convergence}")
    parprint(f"  max steps: {args.steps}")
    if args.fix_cell:
        parprint(f"  mode: fix-cell (positions only - for defects)")
    elif args.fix_c:
        parprint(f"  mode: fix-c (2D/slab)")
    if args.two_stage:
        parprint(f"  two-stage: yes")
    if use_fast:
        parprint(f"  fast mode: yes (coarse → fine)")

    # Dry run - just print info and exit
    if args.dry_run:
        parprint(f"\n[DRY RUN] Would save to:")
        parprint(f"  {output_cif}")
        parprint(f"  {output_xyz}")
        parprint(f"  {gpaw_log}")
        parprint(f"\nExiting (dry run mode).")
        return

    # Set up parallelization
    parallel_opts = pick_kpt_parallel(world.size, kpts)
    if world.size > 1:
        parprint(f"\nMPI parallelization:")
        parprint(f"  Ranks: {world.size}")
        parprint(f"  Parallel options: {parallel_opts}")

    # Symmetry setting
    use_symmetry = "off" if not args.symmetry else {"point_group": True, "time_reversal": True}
    if args.symmetry:
        parprint(f"\nUsing symmetry to reduce k-points (faster)")
        if not args.fix_c:
            parprint(f"  WARNING: Symmetry may break during cell optimization!")

    # =========================================================
    # FAST MODE: Coarse pre-optimization then refine
    # =========================================================
    if use_fast:
        parprint("\n" + "="*50)
        parprint("FAST MODE: Coarse → Fine optimization")
        parprint("="*50)
        
        # Stage 1: Coarse optimization (fast)
        coarse_ecut = min(250, ecut)
        coarse_kpts = tuple(max(1, k//2) for k in kpts)  # Halve k-points
        coarse_kpts = tuple(max(k, 1) for k in coarse_kpts)  # Ensure at least 1
        
        parprint(f"\n[Stage 1/2] COARSE pre-optimization:")
        parprint(f"  ecut: {coarse_ecut} eV (vs final {ecut} eV)")
        parprint(f"  kpts: {coarse_kpts} (vs final {kpts})")
        parprint(f"  convergence: 1e-3 (loose)")
        parprint(f"  target fmax: 0.15 eV/Å")
        
        coarse_parallel = pick_kpt_parallel(world.size, coarse_kpts)
        coarse_log = f"{output_base}_coarse.log"
        if args.lcao:
            coarse_calc = GPAW(
                mode='lcao',
                basis=args.basis,  # Use same basis as fine stage
                xc="PBE",
                occupations=FermiDirac(args.smearing),
                kpts=coarse_kpts,
                txt=coarse_log,
                convergence={"energy": 1e-3, "density": 1e-3},
                symmetry=use_symmetry,
                parallel=coarse_parallel
            )
        else:
            coarse_calc = GPAW(
                mode=PW(coarse_ecut),
                xc="PBE",
                occupations=FermiDirac(args.smearing),
                kpts=coarse_kpts,
                txt=coarse_log,
                convergence={"energy": 1e-3, "density": 1e-3},
                symmetry=use_symmetry,
                parallel=coarse_parallel
            )
        atoms.calc = coarse_calc
        
        # Quick position-only relaxation
        parprint("\n  Running coarse FIRE optimization...")
        coarse_opt = FIRE(atoms, logfile=None)
        coarse_progress = ProgressCallback(atoms, 30, 0.15, label="Coarse FIRE")
        coarse_opt.attach(coarse_progress)
        coarse_opt.run(fmax=0.15, steps=30)
        coarse_progress.close()
        
        e_coarse = atoms.get_potential_energy()
        f_coarse = np.sqrt((atoms.get_forces() ** 2).sum(axis=1)).max()
        parprint(f"\n  Coarse done: E={e_coarse:.4f} eV, fmax={f_coarse:.4f} eV/Å")
        
        # Clear calculator for fine stage
        atoms.calc = None
        
        parprint(f"\n[Stage 2/2] FINE optimization:")
        parprint(f"  ecut: {ecut} eV")
        parprint(f"  kpts: {kpts}")
    
    # =========================================================
    # Create main calculator (or fine-stage calculator in fast mode)
    # =========================================================
    if args.lcao:
        # LCAO mode - faster
        # Note: GPU support in GPAW requires specific compilation and setup
        # It's typically enabled via environment: GPAW_GPU=1 or similar
        if args.gpu:
            parprint(f"\n  Using LCAO mode with '{args.basis}' basis")
            parprint(f"  GPU requested - ensure GPAW is compiled with GPU support")
            parprint(f"  (Set GPAW_GPU=1 or use gpaw-python with GPU build)")
        else:
            parprint(f"\n  Using LCAO mode with '{args.basis}' basis")
        
        calc = GPAW(
            mode='lcao',
            basis=args.basis,
            xc="PBE",
            occupations=FermiDirac(args.smearing),
            kpts=kpts,
            txt=gpaw_log,
            convergence=convergence,
            symmetry="off",  # Must be off for cell optimization
            parallel=parallel_opts
        )
    else:
        # Plane-wave mode - more accurate
        calc = GPAW(
            mode=PW(ecut),
            xc="PBE",
            occupations=FermiDirac(args.smearing),
            kpts=kpts,
            txt=gpaw_log,
            convergence=convergence,
            symmetry="off",  # Must be off for cell optimization
            parallel=parallel_opts
        )
    atoms.calc = calc

    # Skip initial energy calculation - optimizer computes on first step
    # This saves one full SCF cycle (~30s-2min depending on system size)
    parprint("\nStarting optimization (initial energy computed on first step)...")

    # Two-stage optimization
    if args.two_stage:
        parprint("\n=== Stage 1: Position relaxation (cell fixed) ===")
        opt1 = LBFGS(atoms, trajectory=f"{base_name}_stage1.traj", logfile=None)
        stage1_progress = ProgressCallback(atoms, 50, 0.1, label="Position relax")
        opt1.attach(stage1_progress)
        opt1.run(fmax=0.1, steps=50)
        stage1_progress.close()
        parprint(f"  Stage 1 complete. Energy: {atoms.get_potential_energy():.6f} eV")

    # Main optimization (cell + positions)
    # Using FIRE + LBFGS approach (best practice):
    # 1) FIRE for quick pre-relax (robust, handles rough geometries)
    # 2) LBFGS for accurate finish (fast near minimum)
    
    parprint("\n=== Cell + position optimization ===")

    if args.fix_cell:
        # Fix ALL cell parameters - only relax atomic positions
        # This is for defect calculations where supercell size should be fixed
        ecf = atoms  # No cell filter - just optimize positions
        parprint("  Mode: fix-cell (positions only - for defect calculations)")
    elif args.fix_c:
        # Fix c and out-of-plane strains for 2D/slab
        # Mask: [xx, yy, zz, yz, xz, xy] - 1=relax, 0=fix
        strain_mask = [1, 1, 0, 0, 0, 1]  # Allow a, b, and in-plane angle
        ecf = ExpCellFilter(atoms, mask=strain_mask)
        parprint("  Mode: fix-c (2D/slab - optimizing a, b only)")
    else:
        ecf = ExpCellFilter(atoms)
        parprint("  Mode: full cell relaxation")

    if args.fire_only:
        # FIRE only - more robust for defects, light atoms (H), rough PES
        parprint("\n  FIRE-only mode (robust for defects/H atoms)")
        parprint(f"  Target: fmax < {fmax} eV/Å, max {args.steps} steps")
        fire_opt = FIRE(ecf, dt=0.1, maxstep=0.1, trajectory=traj_file, logfile=None)
        fire_progress = ProgressCallback(atoms, args.steps, fmax, label="FIRE")
        fire_opt.attach(fire_progress)
        fire_opt.run(fmax=fmax, steps=args.steps)
        fire_progress.close()
    else:
        # Two-stage: FIRE pre-relax + LBFGS finish (faster for well-behaved systems)
        # Stage 1: FIRE pre-relax (robust for rough geometries)
        parprint("\n  [1/2] FIRE pre-relax (fmax=0.1)...")
        fire_opt = FIRE(ecf, trajectory=f"{base_name}_fire.traj", logfile=None)
        fire_progress = ProgressCallback(atoms, 50, 0.1, label="FIRE pre-relax")
        fire_opt.attach(fire_progress)
        fire_opt.run(fmax=0.1, steps=50)
        fire_progress.close()
        f_after_fire = np.sqrt((atoms.get_forces() ** 2).sum(axis=1)).max()
        parprint(f"        FIRE done. fmax={f_after_fire:.4f} eV/Å")

        # Stage 2: LBFGS finish (fast convergence near minimum)
        parprint(f"\n  [2/2] LBFGS finish (fmax={fmax})...")
        lbfgs_opt = LBFGS(ecf, maxstep=0.04, trajectory=traj_file, logfile=None)
        lbfgs_progress = ProgressCallback(atoms, args.steps, fmax, label="LBFGS finish")
        lbfgs_opt.attach(lbfgs_progress)
        lbfgs_opt.run(fmax=fmax, steps=args.steps)
        lbfgs_progress.close()

    # Final results
    e1 = atoms.get_potential_energy()
    forces = atoms.get_forces()
    fmax1 = np.sqrt((forces ** 2).sum(axis=1)).max()

    parprint(f"\n{'='*50}")
    parprint("OPTIMIZATION COMPLETE")
    parprint(f"{'='*50}")
    parprint(f"  Final energy:   {e1:.6f} eV")
    parprint(f"  Final max force: {fmax1:.6f} eV/Å")

    parprint(f"\nFinal structure:")
    parprint(f"  Cell (a, b, c): {atoms.cell.lengths()}")
    parprint(f"  Angles (α, β, γ): {atoms.cell.angles()}")
    parprint(f"  Volume: {atoms.get_volume():.3f} Å³")

    # Convergence check
    if fmax1 > fmax:
        parprint(f"\n  WARNING: Did not converge to fmax={fmax}!")
        parprint(f"           Current fmax={fmax1:.4f}. Try more steps or two-stage.")

    # Save outputs
    io.write(output_cif, atoms)
    io.write(output_xyz, atoms)
    parprint(f"\nOutput directory: {output_dir}/")
    parprint(f"Saved optimized structure:")
    parprint(f"  {output_cif}")
    parprint(f"  {output_xyz}")
    parprint(f"  {gpaw_log}")
    parprint("\nDone.")


if __name__ == "__main__":
    main()
