#!/bin/bash
#SBATCH --mem=8G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --mail-user=ssg57@sfu.ca
#SBATCH --mail-type=END
#SBATCH --account=def-danielac
#SBATCH --array=1-10

# Usage: sbatch slurm/queue_sac.sh <pendulum|double|hopper|walker>
ENV=${1:-pendulum}

# Code to run (override with: REPO=/path sbatch ...; sbatch passes the variable through)
REPO=${REPO:-$HOME/projects/def-danielac/sarvan13/stable-lppo/scaled-lyapunov-rl}
cd "$REPO" || { echo "REPO not found: $REPO"; exit 1; }
module purge
module load python/3.10.13
module load mujoco
source ~/MujocoENV/bin/activate
source slurm/envs.sh
if [ -z "$OFF_STEPS" ]; then echo "SAC/LSAC are not configured for $ENV"; exit 1; fi

python algorithms/clean_sac.py --env-id $ENV_ID --save-model --total-timesteps $OFF_STEPS --seed $SLURM_ARRAY_TASK_ID
