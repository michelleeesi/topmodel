#!/bin/bash
#SBATCH --job-name=bald_corner
#SBATCH --partition=shared
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=bald_corner_%j.out
#SBATCH --error=bald_corner_%j.err
# Corner-only re-run: just (tau_r, tau_kappa) = (0, 0). 30 (cell, trial) jobs.
# Should finish in 10-30 min on 32 cores.

set -euo pipefail

echo "[$(date)] starting on $(hostname)"
echo "[$(date)] SLURM_JOB_ID=$SLURM_JOB_ID  SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
echo "[$(date)] cwd=$(pwd)"

# --- Activate Python environment (match sbatch_big_sweep_ui.sh) ----------
module load python
source activate bald

python -c "import numpy, scipy, joblib, matplotlib; print('deps OK:', numpy.__version__, scipy.__version__, joblib.__version__, matplotlib.__version__)"

# --- Run --------------------------------------------------------------------
cd "$SLURM_SUBMIT_DIR"
python -u run_corner_only.py

echo "[$(date)] done"
