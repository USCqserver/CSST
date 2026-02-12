#!/bin/bash
#
# USER: Replace the YOUR_* placeholders below with values appropriate for your cluster/user.
#

#SBATCH --account=YOUR_ACCOUNT_OR_ALLOCATION        # USER: your Slurm account (or delete if not required)
#SBATCH --mail-type=all
#SBATCH --mail-user=YOUR_EMAIL@EXAMPLE.COM          # USER: your email (or remove mail lines if undesired)
#SBATCH --partition=debug                           # USER: change partition if needed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --job-name=data_gen

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
NW=$SLURM_CPUS_PER_TASK   # uses whatever you set in --cpus-per-task

# -------- Work dir on SCRATCH (so ./data lands here) --------
SCRATCH="YOUR_SCRATCH_PATH"                         # USER: e.g. "/scratch1/$USER" or "$SCRATCH"
WORKDIR="$SCRATCH/CSST"                              # USER: choose project subdirectory
mkdir -p "$WORKDIR"
cd "$WORKDIR"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# -------- Miniconda: init + activate environment --------
# (Batch shells don't read your ~/.bashrc)
source "YOUR_CONDA_SH_PATH"                         # USER: e.g. "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate YOUR_CONDA_ENV_NAME                  # USER: e.g. "shadows"

# -------- Run your script from HOME; I/O happens in $WORKDIR --------
python "YOUR_PATH_TO_REPO/DATA_GEN.py" \            # USER: e.g. "$HOME/CSST/DATA_GEN.py"
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
