#!/bin/bash
# Submit every algorithm / scaling combination on every environment (each job is a 10-seed array).
#
# Usage (run directly, not with sbatch):
#   bash slurm/queue_all.sh                    # pendulum, double and quad
#   bash slurm/queue_all.sh pendulum double    # only some environments
#   DRY_RUN=1 bash slurm/queue_all.sh          # print the sbatch commands without submitting
#
# SAC and LSAC are skipped on quad (see slurm/envs.sh).

cd "$(dirname "$0")/.."

ENVS=${@:-pendulum double quad}
MIN_LIE_MU=0.1

# Regularization weight per environment (best values from thesis Chapter 3.2)
reg_mu() {
    case "$1" in
        pendulum) echo 0.0005 ;;
        double)   echo 0.01 ;;
        quad)     echo 0.0005 ;;
    esac
}

submit() {
    if [ -n "$DRY_RUN" ]; then
        echo sbatch "$@"
    else
        echo -n "$* -> "
        sbatch "$@"
    fi
}

for ENV in $ENVS; do
    case "$ENV" in
        pendulum|double|quad) ;;
        *) echo "Unknown environment '$ENV' (use pendulum, double or quad)"; exit 1 ;;
    esac

    LYAP="lppo ly"
    if [ "$ENV" != "quad" ]; then
        submit slurm/queue_sac.sh "$ENV"
        LYAP="lsac $LYAP"
    fi
    submit slurm/queue_ppo.sh "$ENV"

    for ALGO in $LYAP; do
        submit slurm/queue_$ALGO.sh "$ENV" none
        submit slurm/queue_$ALGO.sh "$ENV" min_lie "$MIN_LIE_MU"
        submit slurm/queue_$ALGO.sh "$ENV" regularization "$(reg_mu "$ENV")"
        submit slurm/queue_$ALGO.sh "$ENV" structural
    done
done
