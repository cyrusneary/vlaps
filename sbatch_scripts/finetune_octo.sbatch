#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:l40s:2
#SBATCH --job-name=finetune_octo_base_1p5_wrist      # (optional) give your job a name
#SBATCH --output=logs/%x.%j.out  # (optional) stdout → logs/jobname.jobid.out
#SBATCH --error=logs/%x.%j.err   # (optional) stderr → logs/jobname.jobid.err

source ~/.bashrc

cd ~/projects/foundation_planning_agents
source experiments/libero/.venv-octo/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/LIBERO

module load cuda/11.8/cudnn/8.6
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/ai/apps/x86_64/cuda/11.8"

uv run third_party/octo/scripts/finetune_libero.py