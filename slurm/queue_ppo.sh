#!/bin/bash
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=10:00:0
#SBATCH --mail-user=ssg57@sfu.ca
#SBATCH --mail-type=END
#SBATCH --account=def-danielac
#SBATCH --gpus-per-node=1
#SBATCH --array=1-10

# Usage: sbatch slurm/queue_ppo.sh <pendulum|double|quad|hopper|walker>
ENV=${1:-pendulum}

# Code to run (override with: REPO=/path sbatch ...; sbatch passes the variable through)
REPO=${REPO:-$HOME/projects/def-danielac/sarvan13/stable-lppo/scaled-lyapunov-rl}
cd "$REPO" || { echo "REPO not found: $REPO"; exit 1; }
module purge
module load python/3.10.13
module load mujoco
source ~/MujocoENV/bin/activate
source slurm/envs.sh

python algorithms/clean_ppo.py --env-id $ENV_ID $ON_ARGS --save-model --total-timesteps $ON_STEPS --seed $SLURM_ARRAY_TASK_ID
