#!/bin/bash
# Sweep the scaling weight mu over the values used in the thesis (each job is a 10-seed array).
#   LSAC, LPPO: min_lie and regularization sweeps
#   LY:         regularization sweep only
# mu = 0 is not swept: it is the same algorithm as scaling method "none" (queued by queue_all.sh).
#
# Usage (run directly, not with sbatch):
#   bash slurm/queue_sweep.sh                     # pendulum, double and quad
#   bash slurm/queue_sweep.sh double              # only some environments
#   ALGOS="lppo" bash slurm/queue_sweep.sh quad   # only some algorithms
#   DRY_RUN=1 bash slurm/queue_sweep.sh           # print the sbatch commands without submitting
#   FORCE=1 bash slurm/queue_sweep.sh             # also resubmit configs that already have results in data/
#
# SAC/LSAC are skipped on quad (see slurm/envs.sh).

cd "$(dirname "$0")/.."

ENVS=${@:-pendulum double quad}
ALGOS=${ALGOS:-lsac lppo ly}

# Values from the thesis sweeps (Figures 3.3, 3.5, 3.11, 3.12, 3.16, 3.17)
min_lie_mus() {
    case "$1" in
        pendulum) echo 0.01 0.1 0.15 0.25 0.5 0.75 ;;
        double)   echo 0.05 0.1 0.15 0.25 0.5 0.75 ;;
        quad)     echo 0.1 0.5 0.75 1.0 ;;
    esac
}
reg_mus() {
    case "$1" in
        pendulum) echo 0.0001 0.0005 0.001 0.005 0.05 ;;
        double)   echo 0.0001 0.0005 0.001 0.005 0.01 ;;
        quad)     echo 0.0001 0.0005 0.001 0.005 0.01 ;;
    esac
}

submit() {  # algo env method mu
    local tag="${3}_mu$(printf '%g' "$4")"  # same folder name the python scripts use
    if [ -z "$FORCE" ] && [ -d "data/$ENV_ID/clean_$1/$tag" ]; then
        echo "skip (data/$ENV_ID/clean_$1/$tag exists): $*"
        return
    fi
    if [ -n "$DRY_RUN" ]; then
        echo sbatch slurm/queue_$1.sh "$2" "$3" "$4"
    else
        echo -n "$* -> "
        sbatch slurm/queue_$1.sh "$2" "$3" "$4"
    fi
}

for ENV in $ENVS; do
    source slurm/envs.sh  # sets ENV_ID / OFF_STEPS, exits on an unknown environment

    for ALGO in $ALGOS; do
        if [ "$ALGO" = "lsac" ] && [ -z "$OFF_STEPS" ]; then
            continue
        fi
        if [ "$ALGO" != "ly" ]; then
            for MU in $(min_lie_mus "$ENV"); do
                submit "$ALGO" "$ENV" min_lie "$MU"
            done
        fi
        for MU in $(reg_mus "$ENV"); do
            submit "$ALGO" "$ENV" regularization "$MU"
        done
    done
done
