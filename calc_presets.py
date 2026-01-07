"""
Quality presets for DFT calculations.

Usage:
    from calc_presets import get_preset, QUALITY_CHOICES
    
    preset = get_preset('draft')
    calc = GPAW(
        mode=PW(preset['ecut']),
        kpts=preset['kpts'],
        convergence=preset['convergence'],
        ...
    )

Presets:
    - draft:      Fast screening (~5× speedup, ±0.15 eV accuracy)
    - standard:   Normal workflow (default)
    - production: Publication quality (strict convergence)
"""

QUALITY_CHOICES = ['draft', 'standard', 'production']

# Base presets (for bulk/3D systems)
PRESETS_3D = {
    'draft': {
        'ecut': 300,           # eV
        'kpts': (2, 2, 2),
        'kpts_dos': (3, 3, 3), # Slightly denser for DOS
        'convergence': {'energy': 1e-3, 'density': 1e-3},
        'smearing': 0.1,       # eV
        'dos_width': 0.2,      # eV (broader smoothing)
        'dos_npts': 1001,
        'h_opt_steps': 20,     # Max steps for H optimization
        'h_opt_fmax': 0.1,     # Looser force convergence
        'description': 'Fast screening mode (~5× speedup)',
        'uncertainty': '±0.15 eV'
    },
    'standard': {
        'ecut': 500,
        'kpts': (4, 4, 4),
        'kpts_dos': (6, 6, 6),
        'convergence': {'energy': 1e-4, 'density': 1e-4},
        'smearing': 0.05,
        'dos_width': 0.1,
        'dos_npts': 2001,
        'h_opt_steps': 50,
        'h_opt_fmax': 0.03,
        'description': 'Standard calculation',
        'uncertainty': '±0.05 eV'
    },
    'production': {
        'ecut': 600,
        'kpts': (6, 6, 6),
        'kpts_dos': (8, 8, 8),
        'convergence': {'energy': 1e-6, 'density': 1e-6},
        'smearing': 0.02,
        'dos_width': 0.05,
        'dos_npts': 4001,
        'h_opt_steps': 100,
        'h_opt_fmax': 0.01,
        'description': 'Publication quality (strict convergence)',
        'uncertainty': '±0.02 eV'
    }
}

# Presets for 2D/slab systems (reduced kz)
PRESETS_2D = {
    'draft': {
        'ecut': 300,
        'kpts': (2, 2, 1),
        'kpts_dos': (3, 3, 1),
        'convergence': {'energy': 1e-3, 'density': 1e-3},
        'smearing': 0.1,
        'dos_width': 0.2,
        'dos_npts': 1001,
        'h_opt_steps': 20,
        'h_opt_fmax': 0.1,
        'description': 'Fast 2D screening mode',
        'uncertainty': '±0.15 eV'
    },
    'standard': {
        'ecut': 500,
        'kpts': (4, 4, 1),
        'kpts_dos': (6, 6, 1),
        'convergence': {'energy': 1e-4, 'density': 1e-4},
        'smearing': 0.05,
        'dos_width': 0.1,
        'dos_npts': 2001,
        'h_opt_steps': 50,
        'h_opt_fmax': 0.03,
        'description': 'Standard 2D calculation',
        'uncertainty': '±0.05 eV'
    },
    'production': {
        'ecut': 600,
        'kpts': (8, 8, 1),
        'kpts_dos': (12, 12, 1),
        'convergence': {'energy': 1e-6, 'density': 1e-6},
        'smearing': 0.02,
        'dos_width': 0.05,
        'dos_npts': 4001,
        'h_opt_steps': 100,
        'h_opt_fmax': 0.01,
        'description': 'Publication quality 2D',
        'uncertainty': '±0.02 eV'
    }
}


def get_preset(quality='standard', system_type='2D'):
    """
    Get calculation parameters for given quality level.
    
    Args:
        quality: 'draft', 'standard', or 'production'
        system_type: '2D' or '3D'
    
    Returns:
        dict with ecut, kpts, convergence, etc.
    """
    if quality not in QUALITY_CHOICES:
        raise ValueError(f"Unknown quality '{quality}'. Choose from {QUALITY_CHOICES}")
    
    presets = PRESETS_2D if system_type == '2D' else PRESETS_3D
    return presets[quality].copy()


def detect_system_type(atoms):
    """
    Detect if system is 2D (slab with vacuum) or 3D.
    
    Simple heuristic: if c >> a,b and there's vacuum, it's 2D.
    """
    lengths = atoms.cell.lengths()
    a, b, c = lengths
    
    # If c is much larger than a and b, likely 2D with vacuum
    if c > 1.5 * max(a, b):
        return '2D'
    
    # Check for vacuum (large empty region along c)
    z_coords = atoms.positions[:, 2]
    z_range = z_coords.max() - z_coords.min()
    if z_range < 0.5 * c:  # Material takes < 50% of c
        return '2D'
    
    return '3D'


def print_preset_info(preset, quality):
    """Print preset parameters."""
    print(f"\n{'='*50}")
    print(f"QUALITY PRESET: {quality.upper()}")
    print(f"{'='*50}")
    print(f"  {preset['description']}")
    print(f"  Expected uncertainty: {preset['uncertainty']}")
    print(f"\n  Parameters:")
    print(f"    ecut:        {preset['ecut']} eV")
    print(f"    kpts (SCF):  {preset['kpts']}")
    print(f"    kpts (DOS):  {preset['kpts_dos']}")
    print(f"    convergence: {preset['convergence']}")
    print(f"    smearing:    {preset['smearing']} eV")
    print(f"    DOS width:   {preset['dos_width']} eV")
    
    if quality == 'draft':
        print(f"\n  ⚠️  DRAFT MODE - Results are approximate!")
        print(f"      Re-run with --quality standard for final values.")


def get_quality_warning(quality):
    """Get warning text for draft mode."""
    if quality == 'draft':
        return "⚠️ DRAFT MODE - Results approximate (±0.15 eV). Re-run with --quality standard for publication."
    elif quality == 'production':
        return "✓ PRODUCTION MODE - Publication quality settings."
    return ""



