#!/bin/bash
#SBATCH -J gplfr-synth-lc
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=08:00:00
#SBATCH --array=0-24%10
#SBATCH -o demos/synthetic_learning_curve/slurm-%A_%a.out
#SBATCH -e demos/synthetic_learning_curve/slurm-%A_%a.err

# Optional site-specific directives; uncomment and edit as needed.
# #SBATCH -A <account>
# #SBATCH -C <gpu-constraint>

set -euo pipefail
DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${DEMO_DIR}/../.."

# Activate your environment here if needed, for example:
# source .venv/bin/activate

python demos/synthetic_learning_curve/run_learning_curve.py --task-id "${SLURM_ARRAY_TASK_ID}"
