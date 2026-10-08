#!/bin/bash
#SBATCH --account=pengyu-lab
#SBATCH --partition=pengyu-gpu
#SBATCH --qos=medium
#SBATCH --time=72:00:00
#SBATCH --job-name=abss-eval
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=abss_eval_%j.log

set -euo pipefail
cd "${ABSS_ROOT:-$SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH
exec "${PYTHON:-python}" -u "${ABSS_EVAL_ENTRY:-evaluation/score.py}" "$@"
