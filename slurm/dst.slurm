#!/bin/bash
#SBATCH --account=lidar_492
#SBATCH --mail-type=all
#SBATCH --mail-user=jb_772@usc.edu
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --job-name=dst

set -euo pipefail


# -------- Parameters --------
DIR="debug"
NX=2
NY=3
HAM="heis"
ISTATE="+-+-+-"
N=1000
NB=4
NSMIN=50
NSMAX=50000
NSNUM=30
NW=$SLURM_CPUS_PER_TASK

# -------- Work dir on SCRATCH (so ./data lands here) --------
SCRATCH="/scratch1/jb_772"
WORKDIR="$SCRATCH/dst"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# -------- Miniconda: init + activate shadows --------
# (Batch shells don't read your ~/.bashrc)
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate shadows

# -------- Run your script from HOME; I/O happens in $WORKDIR --------
python "$HOME/dynamical_shadow_tomography/DATA_GEN.py" \
    --dir	"$DIR"		\
    --nx 	"$NX"		\
    --ny 	"$NY"		\
    --ham 	"$HAM"		\
    --n		"$N"		\
    --nb 	"$NB"		\
    --nsmin	"$NSMIN"	\
    --nsmax 	"$NSMAX"	\
    --nsnum 	"$NSNUM" 	\
    --istate 	"$ISTATE" 	\
    --nw 	"$NW"
