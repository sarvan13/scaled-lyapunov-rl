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

# Usage: sbatch slurm/queue_sac.sh <pendulum|double>
ENV=${1:-pendulum}

cd ~/projects/def-danielac/sarvan13/scaled-lyapunov-rl
module purge
module load python/3.10.13
module load mujoco
source ~/MujocoENV/bin/activate
source slurm/envs.sh
if [ -z "$OFF_STEPS" ]; then echo "SAC/LSAC are not configured for $ENV"; exit 1; fi

python algorithms/clean_sac.py --env-id $ENV_ID --save-model --total-timesteps $OFF_STEPS --seed $SLURM_ARRAY_TASK_ID
