#!/bin/bash
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=10:00:0
#SBATCH --mail-user=sarvan13@uvic.ca
#SBATCH --mail-type=ALL
#SBATCH --account=def-danielac
#SBATCH --gpus-per-node=1
#SBATCH --array=1-10

# LY with the Lyapunov function on a subset of the observation (torso and joint angles by default).
# Usage: sbatch slurm/queue_ly_subset.sh <hopper|walker> [scaling_method] [scaling_mu]   (default: none)
#   sbatch slurm/queue_ly_subset.sh hopper structural
#   sbatch slurm/queue_ly_subset.sh walker min_lie 0.1
ENV=${1:-hopper}
METHOD=${2:-none}
MU=${3:-0.0}


cd ~/projects/def-danielac/sarvan13/scaled-lyapunov-rl
module purge
module load python/3.10.13
module load mujoco
source ~/MujocoENV/bin/activate
source slurm/envs.sh

python algorithms/clean_ly_subset.py --env-id $ENV_ID $ON_ARGS --save-model --total-timesteps $ON_STEPS --seed $SLURM_ARRAY_TASK_ID --scaling-method $METHOD --scaling-mu $MU
