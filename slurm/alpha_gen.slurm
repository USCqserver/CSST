#!/bin/bash
#
# USER: Replace the placeholders below (YOUR_...) with values for your cluster/user.
#

#SBATCH --account=YOUR_ACCOUNT_OR_ALLOCATION        # USER: your Slurm account/allocation (or delete this line if not needed)
#SBATCH --mail-type=all
#SBATCH --mail-user=YOUR_EMAIL@EXAMPLE.COM          # USER: your email for Slurm notifications (or delete mail lines if undesired)
#SBATCH --partition=main                            # USER: change if your cluster uses a different partition name
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=alpha_gen

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
SCRATCH="YOUR_SCRATCH_PATH"                         # USER: e.g. "/scratch1/$USER" or "$SCRATCH" if your site defines it
WORKDIR="$SCRATCH/CSST"                              # USER: choose where outputs should land
mkdir -p "$WORKDIR"
cd "$WORKDIR"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# -------- Miniconda: init + activate shadows --------
# (Batch shells don't read your ~/.bashrc)
source "YOUR_CONDA_SH_PATH"                         # USER: e.g. "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate YOUR_CONDA_ENV_NAME                  # USER: e.g. "shadows" (must exist in your conda install)

# -------- Run your script from HOME; I/O happens in $WORKDIR --------
python "YOUR_PATH_TO_REPO/ALPHA_GEN.py" \            # USER: e.g. "$HOME/CSST/ALPHA_GEN.py"
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
