#!/bin/bash
#SBATCH --job-name=bald_test_ui
#SBATCH --partition=test
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=bald_test_ui_%j.out
#SBATCH --error=bald_test_ui_%j.err
# Smoke test for the utilize/ignore-only sweep. Tiny config, fast partition.

set -euo pipefail

echo "[$(date)] starting on $(hostname)"
echo "[$(date)] SLURM_JOB_ID=$SLURM_JOB_ID  SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
echo "[$(date)] cwd=$(pwd)"

# --- Activate Python environment (match sbatch_big_sweep_ui.sh) ----------
module load python
# source activate bald   # uncomment if you created a conda env

python -c "import numpy, scipy, joblib; print('deps OK:', numpy.__version__, scipy.__version__, joblib.__version__)"

# --- Run --------------------------------------------------------------------
cd "$SLURM_SUBMIT_DIR"
python -u run_test_sweep_ui.py

echo "[$(date)] done"
