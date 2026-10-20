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

# Usage: sbatch slurm/queue_ly.sh <pendulum|double|quad> [scaling_method] [scaling_mu]
# Defaults to the unscaled LY baseline (scaling_method none).
ENV=${1:-pendulum}
METHOD=${2:-none}
MU=${3:-0.0}

cd ~/projects/def-danielac/sarvan13/scaled-lyapunov-rl
module purge
module load python/3.10.13
module load mujoco
source ~/MujocoENV/bin/activate
source slurm/envs.sh

python algorithms/clean_ly.py --env-id $ENV_ID $EQ_ARGS $ON_ARGS --save-model --total-timesteps $ON_STEPS --seed $SLURM_ARRAY_TASK_ID --scaling-method $METHOD --scaling-mu $MU
