"""
Geometry optimization script for CIF file using GPAW.

Usage:
    gpaw python optimize_geometry.py input.cif
    mpirun -n 4 gpaw python -- optimize_geometry.py input.cif [options]

Options:
    --ecut       Plane-wave cutoff in eV (default: 500)
    --kpts       k-point grid, e.g. 4 4 4 (default: 4 4 4)
    --smearing   Fermi-Dirac width in eV (default: 0.05)
    --fmax       Force convergence in eV/Å (default: 0.03)
    --steps      Max optimization steps (default: 200)
    --fix-c      Fix c parameter (for 2D/slab with vacuum along c)
    --two-stage  First relax positions, then cell+positions
    --supercell  Create supercell before optimization (e.g., 2 2 1)
    --dry-run    Print structure info only, don't run calculation
    --fast       Coarse pre-optimization then refine (RECOMMENDED for good initial geometries)
    --symmetry   Use symmetry to reduce k-points (faster, but not for cell optimization)

Example:
    # Fast optimization of well-optimized structure
    mpirun -n 4 gpaw python -- optimize_geometry.py g_c3n4.cif --fast --fix-c --kpts 4 4 1

    # Full optimization
    mpirun -n 4 gpaw python -- optimize_geometry.py C3N4.cif --ecut 500 --fmax 0.03
    
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
    GPAW requires:
    1. kpt groups must divide number of k-points
    2. world_size must be divisible by kpt groups
    """
    nk = int(np.prod(kpts))
    if world_size <= 1:
        return None

    # Find largest value that divides BOTH nk and world_size
    # This ensures valid k-point parallelization
    valid_kpt_groups = [d for d in range(1, world_size + 1) 
                        if nk % d == 0 and world_size % d == 0]
    kpt_groups = max(valid_kpt_groups) if valid_kpt_groups else 1

    if kpt_groups == 1:
        # No useful k-point parallelism, let GPAW decide
        return None

    return {'kpt': kpt_groups, 'domain': 1, 'band': 1}


def main():
    parser = argparse.ArgumentParser(
        description="Geometry optimization using GPAW",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("cif_file", nargs="?", default="C3N4_mp-1193580_primitive.cif.txt",
                        help="Input CIF file")
    parser.add_argument("--ecut", type=float, default=500.0,
                        help="Plane-wave cutoff in eV (default: 500)")
    parser.add_argument("--kpts", type=int, nargs=3, default=[4, 4, 4],
                        help="k-point grid (default: 4 4 4)")
    parser.add_argument("--smearing", type=float, default=0.05,
                        help="Fermi-Dirac width in eV (default: 0.05)")
    parser.add_argument("--fmax", type=float, default=0.03,
                        help="Force convergence in eV/Å (default: 0.03)")
    parser.add_argument("--steps", type=int, default=200,
                        help="Max optimization steps (default: 200)")
    parser.add_argument("--fix-c", action="store_true",
                        help="Fix c parameter (for 2D/slab with vacuum along c)")
    parser.add_argument("--two-stage", action="store_true",
                        help="Two-stage: first positions only, then cell+positions")
    parser.add_argument("--supercell", type=int, nargs=3, default=None,
                        metavar=("NX", "NY", "NZ"),
                        help="Create supercell before optimization (e.g., 2 2 1)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print structure info only, don't run calculation")
    parser.add_argument("--fast", action="store_true",
                        help="Fast mode: coarse pre-optimization then refine (recommended for good initial geometries)")
    parser.add_argument("--symmetry", action="store_true",
                        help="Use symmetry to reduce k-points (faster, but incompatible with cell optimization)")
    args = parser.parse_args()

    cif_file = args.cif_file
    kpts = tuple(args.kpts)

    # Generate output filenames
    base_name = os.path.splitext(os.path.basename(cif_file))[0]
    if base_name.endswith(".cif"):
        base_name = base_name[:-4]

    output_cif = f"{base_name}_optimized.cif"
    output_xyz = f"{base_name}_optimized.xyz"
    gpaw_log = f"{base_name}_gpaw.log"
    traj_file = f"{base_name}_optimization.traj"

    # Read structure
    parprint(f"Reading structure from {cif_file}...")
    atoms = io.read(cif_file, format="cif")

    # Create supercell if requested
    if args.supercell is not None:
        sc = tuple(args.supercell)
        n_orig = len(atoms)
        atoms = atoms * sc
        parprint(f"Created {sc[0]}×{sc[1]}×{sc[2]} supercell: {n_orig} → {len(atoms)} atoms")
        # Update base name to include supercell info
        base_name = f"{base_name}_{sc[0]}x{sc[1]}x{sc[2]}"
        output_cif = f"{base_name}_optimized.cif"
        output_xyz = f"{base_name}_optimized.xyz"
        gpaw_log = f"{base_name}_gpaw.log"
        traj_file = f"{base_name}_optimization.traj"

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
    parprint(f"  ecut: {args.ecut} eV")
    parprint(f"  kpts: {kpts} ({np.prod(kpts)} k-points)")
    parprint(f"  smearing: {args.smearing} eV")
    parprint(f"  fmax: {args.fmax} eV/Å")
    parprint(f"  max steps: {args.steps}")
    if args.fix_c:
        parprint(f"  mode: fix-c (2D/slab)")
    if args.two_stage:
        parprint(f"  two-stage: yes")

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
    if args.fast:
        parprint("\n" + "="*50)
        parprint("FAST MODE: Coarse → Fine optimization")
        parprint("="*50)
        
        # Stage 1: Coarse optimization (fast)
        coarse_ecut = min(300, args.ecut)
        coarse_kpts = tuple(max(1, k//2) for k in kpts)  # Halve k-points
        coarse_kpts = tuple(max(k, 1) for k in coarse_kpts)  # Ensure at least 1
        
        parprint(f"\n[Stage 1/2] COARSE pre-optimization:")
        parprint(f"  ecut: {coarse_ecut} eV (vs final {args.ecut} eV)")
        parprint(f"  kpts: {coarse_kpts} (vs final {kpts})")
        parprint(f"  convergence: 1e-3 (loose)")
        parprint(f"  target fmax: 0.1 eV/Å")
        
        coarse_parallel = pick_kpt_parallel(world.size, coarse_kpts)
        coarse_calc = GPAW(
            mode=PW(coarse_ecut),
            xc="PBE",
            occupations=FermiDirac(args.smearing),
            kpts=coarse_kpts,
            txt=f"{base_name}_coarse.log",
            convergence={"energy": 1e-3, "density": 1e-3},
            symmetry=use_symmetry,
            parallel=coarse_parallel
        )
        atoms.calc = coarse_calc
        
        # Quick position-only relaxation
        parprint("\n  Running coarse FIRE optimization...")
        coarse_opt = FIRE(atoms, logfile=None)
        coarse_progress = ProgressCallback(atoms, 30, 0.1, label="Coarse FIRE")
        coarse_opt.attach(coarse_progress)
        coarse_opt.run(fmax=0.1, steps=30)
        coarse_progress.close()
        
        e_coarse = atoms.get_potential_energy()
        f_coarse = np.sqrt((atoms.get_forces() ** 2).sum(axis=1)).max()
        parprint(f"\n  Coarse done: E={e_coarse:.4f} eV, fmax={f_coarse:.4f} eV/Å")
        
        # Clear calculator for fine stage
        atoms.calc = None
        
        parprint(f"\n[Stage 2/2] FINE optimization:")
        parprint(f"  ecut: {args.ecut} eV")
        parprint(f"  kpts: {kpts}")
    
    # =========================================================
    # Create main calculator (or fine-stage calculator in fast mode)
    # =========================================================
    calc = GPAW(
        mode=PW(args.ecut),
        xc="PBE",
        occupations=FermiDirac(args.smearing),
        kpts=kpts,
        txt=gpaw_log,
        convergence={"energy": 1e-4, "density": 1e-4},
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

    if args.fix_c:
        # Fix c and out-of-plane strains for 2D/slab
        # Mask: [xx, yy, zz, yz, xz, xy] - 1=relax, 0=fix
        strain_mask = [1, 1, 0, 0, 0, 1]  # Allow a, b, and in-plane angle
        ecf = ExpCellFilter(atoms, mask=strain_mask)
        parprint("  Mode: fix-c (2D/slab - optimizing a, b only)")
    else:
        ecf = ExpCellFilter(atoms)
        parprint("  Mode: full cell relaxation")

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
    parprint(f"\n  [2/2] LBFGS finish (fmax={args.fmax})...")
    lbfgs_opt = LBFGS(ecf, trajectory=traj_file, logfile=None)
    lbfgs_progress = ProgressCallback(atoms, args.steps, args.fmax, label="LBFGS finish")
    lbfgs_opt.attach(lbfgs_progress)
    lbfgs_opt.run(fmax=args.fmax, steps=args.steps)
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
    if fmax1 > args.fmax:
        parprint(f"\n  WARNING: Did not converge to fmax={args.fmax}!")
        parprint(f"           Current fmax={fmax1:.4f}. Try more steps or two-stage.")

    # Save outputs
    io.write(output_cif, atoms)
    io.write(output_xyz, atoms)
    parprint(f"\nSaved optimized structure:")
    parprint(f"  {output_cif}")
    parprint(f"  {output_xyz}")
    parprint("\nDone.")


if __name__ == "__main__":
    main()
