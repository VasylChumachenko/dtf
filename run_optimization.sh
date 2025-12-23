#!/bin/bash
# Helper script to run geometry optimization
# Uses mpirun (Open MPI) with gpaw python wrapper

cd /home/vasylchumachenko/dft
source dftenv/bin/activate

# Default values
NPROCS=1
CIF_FILE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--nprocs)
            NPROCS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [-n NPROCS] [cif_file]"
            echo "  -n N        Number of MPI processes (default: 1 = serial)"
            echo "  cif_file    Input CIF file (default: C3N4_mp-1193580_primitive.cif.txt)"
            echo ""
            echo "Examples:"
            echo "  $0                              # Serial, default CIF"
            echo "  $0 my_structure.cif             # Serial, custom CIF"
            echo "  $0 -n 4 my_structure.cif        # 4 MPI processes, custom CIF"
            exit 0
            ;;
        -*)
            echo "Unknown option: $1"
            exit 1
            ;;
        *)
            CIF_FILE="$1"
            shift
            ;;
    esac
done

# Run calculation
if [ "$NPROCS" -eq 1 ]; then
    echo "Running serial..."
    if [ -n "$CIF_FILE" ]; then
        gpaw python optimize_geometry.py "$CIF_FILE"
    else
        gpaw python optimize_geometry.py
    fi
else
    echo "Running with $NPROCS MPI processes..."
    if [ -n "$CIF_FILE" ]; then
        mpirun -n $NPROCS gpaw python optimize_geometry.py "$CIF_FILE"
    else
        mpirun -n $NPROCS gpaw python optimize_geometry.py
    fi
fi
