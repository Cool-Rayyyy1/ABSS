#!/bin/bash
#SBATCH --account=pengyu-lab
#SBATCH --partition=pengyu-gpu
#SBATCH --qos=medium
#SBATCH --time=72:00:00
#SBATCH --job-name=abss
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=abss_%A_%a.log

set -euo pipefail
cd "${ABSS_ROOT:-$SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
unset PYTHONPATH
exec "${PYTHON:-python}" -u "${ABSS_ENTRY:-run.py}" "$@"
