#!/bin/bash
#SBATCH --mail-type=all
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=5:00:00
#SBATCH --job-name=alpha_gen
#
# --account and --mail-user come from SBATCH_ACCOUNT / SBATCH_MAIL_USER
# env vars, exported from ~/.bashrc at submit time (see config.sh.example).

set -euo pipefail

SLURM_DIR="$HOME/CSST/slurm"   # USER: change if your repo lives elsewhere
source "$SLURM_DIR/config.sh"

# -------- Parameters ---------
DIR="data"
NX=2
NY=4
HAM="tfim"
ISTATE="ghz"
EPS=0.1
MMIN=50
MMAX=1000
MNUM=30
AMIN=-7
AMAX=-2
ANUM=40
NW=64
SNR=-1
RESCALE=0
AXIS=1
INVERSE=0
FITINT=1
REPLACE=0

# -------- Work dir on SCRATCH (so ./data lands here) --------
WORKDIR="$SCRATCH_ROOT/dst"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# -------- Miniconda: init + activate env --------
# (Batch shells don't read your ~/.bashrc)
source "$CONDA_SH"
conda activate "$CONDA_ENV"

# -------- Run script from repo; I/O happens in $WORKDIR --------
python "$REPO_PATH/ALPHA_GEN.py" \
    --dir	"$DIR"		\
    --nx	"$NX"		\
    --ny	"$NY"		\
    --ham 	"$HAM" 		\
    --istate 	"$ISTATE" 	\
    --eps	"$EPS"		\
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
