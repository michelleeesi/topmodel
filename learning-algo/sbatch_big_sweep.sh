#!/bin/bash
#SBATCH --job-name=bald_sweep
#SBATCH --partition=shared
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=bald_sweep_%j.out
#SBATCH --error=bald_sweep_%j.err
# %j is the slurm job ID; logs land next to this script.

set -euo pipefail

echo "[$(date)] starting on $(hostname)"
echo "[$(date)] SLURM_JOB_ID=$SLURM_JOB_ID  SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
echo "[$(date)] cwd=$(pwd)"

# --- Activate Python environment -----------------------------------------
# Adjust ONE of these blocks to match your cluster setup. If you don't yet
# have a Python env on the cluster, follow the "first-time setup" notes below.

# Option A: anaconda module + your conda env (most common on FAS)
module load python
# source activate bald   # uncomment and replace `bald` with your env name

# Option B: explicit miniconda / mamba install you set up yourself
# source $HOME/miniconda3/etc/profile.d/conda.sh
# conda activate bald

# Option C: plain venv
# source $HOME/envs/bald/bin/activate

python -c "import numpy, scipy, joblib; print('deps OK:', numpy.__version__, scipy.__version__, joblib.__version__)"

# --- Run --------------------------------------------------------------------
cd "$SLURM_SUBMIT_DIR"
python -u run_big_sweep.py

echo "[$(date)] done"
