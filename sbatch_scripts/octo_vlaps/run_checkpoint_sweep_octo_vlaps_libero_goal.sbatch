#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-2%1                       # 20 tasks, max 1 running at once
#SBATCH --job-name=libero_goal_vlaps_octo_ckpt_%A_%a
#SBATCH --output=logs/%x_%A_%a.out           # e.g. logs/libero_octo_ckpt_1234567_03.out
#SBATCH --error=logs/%x_%A_%a.err

# ------------------------------------------------------------------------
# 1. Map this array index (0-2) to the real checkpoint number.
# ------------------------------------------------------------------------
CHECKPOINTS=(100000 150000 200000)
CKPT_NUM=${CHECKPOINTS[$SLURM_ARRAY_TASK_ID]}

# ------------------------------------------------------------------------
# 2. Environment setup.
# ------------------------------------------------------------------------
source ~/.bashrc
cd ~/projects/foundation_planning_agents
source experiments/libero/.venv-octo/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/LIBERO

module load cuda/11.8/cudnn/8.6
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/ai/apps/x86_64/cuda/11.8"

# ------------------------------------------------------------------------
# 3. Launch the experiment, overriding the checkpoint number
#    and giving each run its own Hydra directory.
# ------------------------------------------------------------------------
# Launch experiment with Hydra config overrides
uv run experiments/libero/run_libero_experiment.py \
  --config-name=config_vlaps_octo.yaml \
  runner_config=libero_goal\
  env_config=libero_goal\
  agent_config=vlaps_octo_agent_config\
  agent_config.vla_config.checkpoint="${CKPT_NUM}" \
  experiment_name="octo_vlaps_goal_ckpt${CKPT_NUM}" \
  agent_config.vla_config.dataset_statistics=libero_goal_no_noops