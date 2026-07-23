#!/bin/bash
#SBATCH --mail-type=all
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --job-name=data_gen
#
# --account and --mail-user come from SBATCH_ACCOUNT / SBATCH_MAIL_USER
# env vars, exported from ~/.bashrc at submit time (see config.sh.example).

set -euo pipefail

SLURM_DIR="$HOME/CSST/slurm"   # USER: change if your repo lives elsewhere
source "$SLURM_DIR/config.sh"

# -------- Parameters --------
DIR="data"
NX=2
NY=2
HAM="heis"
ISTATE="+-+-"
N=1000
NB=4
NSMIN=50
NSMAX=50000
NSNUM=30
TPLIMIT=1
NW=$((SLURM_CPUS_PER_TASK / TPLIMIT))
EPS=0.1

# -------- Work dir on SCRATCH (so ./data lands here) --------
WORKDIR="$SCRATCH_ROOT/dst"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

# NOTE: workers override these via threadpool_limits(limits=TPLIMIT) in
# their init function, regardless of what's exported here.
export OMP_NUM_THREADS=$TPLIMIT
export MKL_NUM_THREADS=$TPLIMIT
export OPENBLAS_NUM_THREADS=$TPLIMIT
export NUMEXPR_NUM_THREADS=$TPLIMIT

# -------- Miniconda: init + activate env --------
# (Batch shells don't read your ~/.bashrc)
source "$CONDA_SH"
conda activate "$CONDA_ENV"

# -------- Warn on thread oversubscription --------
TOTAL_THREADS=$((NW * TPLIMIT))
if [ "$TOTAL_THREADS" -gt "$SLURM_CPUS_PER_TASK" ]; then
    echo "WARNING: NW ($NW) * TPLIMIT ($TPLIMIT) = $TOTAL_THREADS threads requested, exceeding the $SLURM_CPUS_PER_TASK allocated CPUs. This will oversubscribe the node."
fi

# -------- Run script from repo; I/O happens in $WORKDIR --------
python "$REPO_PATH/data_gen.py" \
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
    --nw 	"$NW"		\
    --tplimit	"$TPLIMIT"	\
    --eps	"$EPS"
