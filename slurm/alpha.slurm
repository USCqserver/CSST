#!/bin/bash
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --mail-type=all
#SBATCH --mail-user=YOUR_EMAIL
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=alpha_dst

set -euo pipefail

# -------- Parameters ---------
NX=2
NY=3
HAM="heis"
ISTATE="+-+-+-"
MMIN=50
MMAX=1000
MNUM=30
AMIN=-7
AMAX=0
ANUM=50
NW=64
SNR=-1
RESCALE=1
AXIS=1
INVERSE=0
FITINT=1
REPLACE=0

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
python "$HOME/dynamical_shadow_tomography/ALPHA_GEN.py" \
    --nx	"$NX"		\
    --ny	"$NY"		\
    --ham 	"$HAM" 		\
    --istate 	"$ISTATE" 	\
    --mmin	"$MMIN"		\
    --mmax	"$MMAX"		\
    --mnum	"$MNUM"		\
    --amin	"$AMIN"		\
    --amax	"$AMAX"		\
    --anum	"$ANUM"		\
    --nw	"$NW"		\
    --snr	"$SNR"		\
    --rescale	"$RESCALE"	\
    --axis	"$AXIS"		\
    --inverse	"$INVERSE"	\
    --fitint	"$FITINT"	\
    --replace	"$REPLACE"

