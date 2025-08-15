#!/bin/bash
#SBATCH --partition long           # choose the “unkillable” partition
#SBATCH -c 6                     # reserve 6 CPU cores
#SBATCH --mem=32G                # reserve 32 GB RAM
#SBATCH --gres=gpu:a100:1             # reserve 1 GPU
#SBATCH --job-name=libero_octo_goal    # (optional) give your job a name
#SBATCH --output=logs/%x.%j.out  # (optional) stdout → logs/jobname.jobid.out
#SBATCH --error=logs/%x.%j.err   # (optional) stderr → logs/jobname.jobid.err

source ~/.bashrc

cd ~/projects/foundation_planning_agents
source experiments/libero/.venv-octo/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/LIBERO

module load cuda/11.8/cudnn/8.6
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/ai/apps/x86_64/cuda/11.8"

# Launch experiment with Hydra config overrides
uv run experiments/libero/run_libero_experiment.py \
  --config-name=config_octo.yaml \
  runner_config=libero_goal\
  env_config=libero_goal\
  agent_config=octo_agent\
  experiment_name=octo_goal\
  agent_config.vla_config.dataset_statistics=libero_goal_no_noops