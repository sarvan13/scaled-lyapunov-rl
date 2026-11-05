#!/bin/bash
#SBATCH --mem=8G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --mail-user=ssg57@sfu.ca
#SBATCH --mail-type=END
#SBATCH --account=def-danielac
#SBATCH --array=1-10

# Usage: sbatch slurm/queue_lppo.sh <pendulum|double|quad> <scaling_method> [scaling_mu]
#   sbatch slurm/queue_lppo.sh quad structural
#   sbatch slurm/queue_lppo.sh quad min_lie 0.75
#   sbatch slurm/queue_lppo.sh pendulum regularization 0.0005
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

python algorithms/clean_lppo.py --env-id $ENV_ID $EQ_ARGS $ON_ARGS --save-model --total-timesteps $ON_STEPS --seed $SLURM_ARRAY_TASK_ID --scaling-method $METHOD --scaling-mu $MU
