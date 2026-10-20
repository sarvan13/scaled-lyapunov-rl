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

# Usage: sbatch slurm/queue_lppo.sh <pendulum|double|quad> <scaling_method> [scaling_mu]
#   sbatch slurm/queue_lppo.sh quad structural
#   sbatch slurm/queue_lppo.sh quad min_lie 0.75
#   sbatch slurm/queue_lppo.sh pendulum regularization 0.0005
ENV=${1:-pendulum}
METHOD=${2:-min_lie}
MU=${3:-0.1}

cd ~/projects/def-danielac/sarvan13/scaled-lyapunov-rl
module purge
module load python/3.10.13
module load mujoco
source ~/MujocoENV/bin/activate
source slurm/envs.sh

python algorithms/clean_lppo.py --env-id $ENV_ID $EQ_ARGS $ON_ARGS --save-model --total-timesteps $ON_STEPS --seed $SLURM_ARRAY_TASK_ID --scaling-method $METHOD --scaling-mu $MU
