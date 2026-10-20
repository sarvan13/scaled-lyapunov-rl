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

# Usage: sbatch slurm/queue_lsac.sh <pendulum|double> <scaling_method> [scaling_mu]
#   sbatch slurm/queue_lsac.sh pendulum min_lie 0.1
#   sbatch slurm/queue_lsac.sh double regularization 0.01
#   sbatch slurm/queue_lsac.sh double structural
#   sbatch slurm/queue_lsac.sh pendulum none
ENV=${1:-pendulum}
METHOD=${2:-min_lie}
MU=${3:-0.1}

case "$ENV" in
    pendulum)
        ENV_ID=InvertedPendulum-v5
        EQ_ARGS=""  # obs = [x, theta, v, omega]; upright is all zeros
        ;;
    double)
        ENV_ID=InvertedDoublePendulum-v5
        EQ_ARGS="--eq-obs 0 0 0 1 1 0 0 0 0"  # obs = [x, sin t1, sin t2, cos t1, cos t2, v, w1, w2, F_c]
        ;;
    *)
        echo "Unknown environment '$ENV' (use pendulum or double)"
        exit 1
        ;;
esac

cd ~/projects/def-danielac/sarvan13/scaled-lyapunov-rl
module purge
module load python/3.10.13
module load mujoco
source ~/MujocoENV/bin/activate

python algorithms/clean_lsac.py --env-id $ENV_ID $EQ_ARGS --save-model --total-timesteps 500000 --seed $SLURM_ARRAY_TASK_ID --scaling-method $METHOD --scaling-mu $MU
