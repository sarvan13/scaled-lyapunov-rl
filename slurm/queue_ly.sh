#!/bin/bash
#SBATCH --mem=8G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --mail-user=ssg57@sfu.ca
#SBATCH --mail-type=END
#SBATCH --account=def-danielac
#SBATCH --array=1-10

# Usage: sbatch slurm/queue_ly.sh <pendulum|double|quad> [scaling_method] [scaling_mu]
# Defaults to the unscaled LY baseline (scaling_method none).
ENV=${1:-pendulum}
METHOD=${2:-none}
MU=${3:-0.0}

# Code to run (override with: REPO=/path sbatch ...; sbatch passes the variable through)
REPO=${REPO:-$HOME/projects/def-danielac/sarvan13/stable-lppo/scaled-lyapunov-rl}
cd "$REPO" || { echo "REPO not found: $REPO"; exit 1; }
module purge
module load python/3.10.13
module load mujoco
source ~/MujocoENV/bin/activate
source slurm/envs.sh

python algorithms/clean_ly.py --env-id $ENV_ID $EQ_ARGS $ON_ARGS --save-model --total-timesteps $ON_STEPS --seed $SLURM_ARRAY_TASK_ID --scaling-method $METHOD --scaling-mu $MU
