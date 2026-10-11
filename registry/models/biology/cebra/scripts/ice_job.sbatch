#!/bin/bash
#SBATCH --job-name=cebra-achilles
#SBATCH --account=coc
#SBATCH --qos=coc-ice
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

if ! type module >/dev/null 2>&1; then
  source /etc/profile.d/modules.sh
fi

module purge
module load python/3.12.5
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs results

source "${HOME}/scratch/cebra-venv/bin/activate"
nvidia-smi -L
python -V
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
python -m pip freeze > requirements-freeze.txt
python scripts/smoke_test.py
python scripts/run_hippocampus.py
