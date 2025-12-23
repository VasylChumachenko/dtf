#!/bin/bash
# Helper script to run geometry optimization
# Uses mpirun (Open MPI) with gpaw python wrapper

cd /home/vasylchumachenko/dft
source dftenv/bin/activate

# Default values
NPROCS=1
SCRIPT_ARGS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--nprocs)
            NPROCS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [-n NPROCS] [script_args...]"
            echo ""
            echo "Options:"
            echo "  -n N    Number of MPI processes (default: 1 = serial)"
            echo ""
            echo "Script arguments (passed to optimize_geometry.py):"
            echo "  input.cif           Input CIF file"
            echo "  --ecut N            Plane-wave cutoff in eV (default: 500)"
            echo "  --kpts X Y Z        k-point grid (default: 4 4 4)"
            echo "  --smearing N        Fermi-Dirac width in eV (default: 0.05)"
            echo "  --fmax N            Force convergence in eV/Å (default: 0.03)"
            echo "  --steps N           Max optimization steps (default: 200)"
            echo "  --fix-c             Fix c parameter (for 2D/slab)"
            echo "  --two-stage         Two-stage: positions first, then cell"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Serial, defaults"
            echo "  $0 -n 4 C3N4.cif                      # 4 MPI, custom CIF"
            echo "  $0 -n 4 slab.cif --fix-c --kpts 6 6 1 # 2D slab"
            echo "  $0 -n 8 struct.cif --ecut 600 --fmax 0.02"
            exit 0
            ;;
        -*)
            # Unknown option starting with - goes to script
            SCRIPT_ARGS="$SCRIPT_ARGS $1"
            shift
            ;;
        *)
            # Positional args go to script
            SCRIPT_ARGS="$SCRIPT_ARGS $1"
            shift
            ;;
    esac
done

# Run calculation
if [ "$NPROCS" -eq 1 ]; then
    echo "Running serial..."
    echo "Command: gpaw python -- optimize_geometry.py $SCRIPT_ARGS"
    echo ""
    gpaw python -- optimize_geometry.py $SCRIPT_ARGS
else
    echo "Running with $NPROCS MPI processes..."
    echo "Command: mpirun -n $NPROCS gpaw python -- optimize_geometry.py $SCRIPT_ARGS"
    echo ""
    mpirun -n $NPROCS gpaw python -- optimize_geometry.py $SCRIPT_ARGS
fi
