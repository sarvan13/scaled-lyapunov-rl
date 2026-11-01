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

# Usage: sbatch slurm/queue_lsac.sh <pendulum|double> <scaling_method> [scaling_mu]
#   sbatch slurm/queue_lsac.sh pendulum min_lie 0.1
#   sbatch slurm/queue_lsac.sh double regularization 0.01
#   sbatch slurm/queue_lsac.sh double structural
ENV=${1:-pendulum}
METHOD=${2:-min_lie}
MU=${3:-0.1}

# Code to run (override with: REPO=/path sbatch ...; sbatch passes the variable through)
REPO=${REPO:-$HOME/projects/def-danielac/sarvan13/stable-lppo/scaled-lyapunov-rl}
cd "$REPO" || { echo "REPO not found: $REPO"; exit 1; }
module purge
module load python/3.10.13
module load mujoco
source ~/MujocoENV/bin/activate
source slurm/envs.sh
if [ -z "$OFF_STEPS" ]; then echo "SAC/LSAC are not configured for $ENV"; exit 1; fi

python algorithms/clean_lsac.py --env-id $ENV_ID $EQ_ARGS --save-model --total-timesteps $OFF_STEPS --seed $SLURM_ARRAY_TASK_ID --scaling-method $METHOD --scaling-mu $MU
