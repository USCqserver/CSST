#!/bin/bash
#SBATCH --mail-type=all
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --job-name=sparsity
#
# --account and --mail-user are supplied at submit time by ./submit.sh,
# which reads them from config.sh (see config.sh.example).

set -euo pipefail

SLURM_DIR="$HOME/CSST/slurm"   # USER: change if your repo lives elsewhere
source "$SLURM_DIR/config.sh"

# -------- Parameters --------
DIR="data"
NX=2
NY=4
NB=4
HAM="heis"
ISTATE="+-+-+-+-"
TRIALS=10
EPS=0.1
EPSMIN=0.0
EPSMAX=0.3
EPSNUM=13
GAMMA=1e-2
NW=$SLURM_CPUS_PER_TASK

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
python "$REPO_PATH/SPARSITY.py" \
    --dir 	"$DIR"		\
    --nx 	"$NX"		\
    --ny 	"$NY"		\
    --nb 	"$NB"		\
    --ham 	"$HAM"		\
    --istate 	"$ISTATE" 	\
    --trials 	"$TRIALS"	\
    --eps	"$EPS"		\
    --epsmin	"$EPSMIN"	\
    --epsmax	"$EPSMAX"	\
    --epsnum	"$EPSNUM"	\
    --gamma	"$GAMMA"	\
    --nw 	"$NW"
