#!/bin/bash
# Queue experiments on quad, hopper and walker (each job is a 10-seed array).
#
#   quad:           LPPO and LY with every scaling method
#   hopper, walker: LY (clean_ly_subset.py) with every scaling method, plus PPO and SAC baselines
#
# Scaling methods run for each Lyapunov algorithm:
#   none            baseline
#   structural
#   min_lie         mu in MIN_LIE_MUS
#   regularization  mu in REG_MUS
#
# Usage (run directly on the server, not with sbatch):
#   bash slurm/queue_quad_hopper_walker.sh                  # quad, hopper and walker
#   bash slurm/queue_quad_hopper_walker.sh hopper walker    # only some environments
#   ALGOS="ly" bash slurm/queue_quad_hopper_walker.sh       # only some algorithms (lppo, ly, ppo, sac)
#   DRY_RUN=1 bash slurm/queue_quad_hopper_walker.sh        # print the sbatch commands without submitting
#   FORCE=1 bash slurm/queue_quad_hopper_walker.sh          # also resubmit configs that already have results in data/

cd "$(dirname "$0")/.."

ENVS=${@:-quad hopper walker}

MIN_LIE_MUS="0.05 0.1 0.5 0.75"
REG_MUS="0.0001 0.0005 0.001 0.005 0.01"

# Algorithms run on each environment
env_algos() {
    case "$1" in
        quad)          echo lppo ly ;;
        hopper|walker) echo ly ppo sac ;;
    esac
}

wanted() {  # true if ALGOS is unset or contains $1
    [ -z "$ALGOS" ] || [[ " $ALGOS " == *" $1 "* ]]
}

count=0
skipped=0

submit() {  # script env [method mu]
    local script=$1 env=$2 method=$3 mu=$4

    # Results folder the python scripts write to (run_tag in algorithms/lyapunov_scaling.py)
    local dir="data/$ENV_ID/clean_$script"
    case "$method" in
        min_lie|regularization) dir="$dir/${method}_mu$(printf '%g' "$mu")" ;;
        none|structural)        dir="$dir/$method" ;;
    esac
    if [ -z "$FORCE" ] && [ -d "$dir" ]; then
        echo "skip ($dir exists): $script $env $method $mu"
        skipped=$((skipped + 1))
        return
    fi

    if [ -n "$DRY_RUN" ]; then
        echo sbatch slurm/queue_$script.sh "$env" $method $mu
    else
        echo -n "$script $env $method $mu -> "
        sbatch slurm/queue_$script.sh "$env" $method $mu
    fi
    count=$((count + 1))
}

submit_scaled() {  # script env: every scaling method
    submit "$1" "$2" none 0.0
    submit "$1" "$2" structural 0.0
    for MU in $MIN_LIE_MUS; do
        submit "$1" "$2" min_lie "$MU"
    done
    for MU in $REG_MUS; do
        submit "$1" "$2" regularization "$MU"
    done
}

for ENV in $ENVS; do
    case "$ENV" in
        quad|hopper|walker) ;;
        *) echo "Unknown environment '$ENV' (use quad, hopper or walker)"; exit 1 ;;
    esac
    source slurm/envs.sh  # sets ENV_ID

    for ALGO in $(env_algos "$ENV"); do
        wanted "$ALGO" || continue
        case "$ALGO" in
            lppo)    submit_scaled lppo "$ENV" ;;
            ly)      if [ "$ENV" = "quad" ]; then submit_scaled ly "$ENV"; else submit_scaled ly_subset "$ENV"; fi ;;
            ppo|sac) submit "$ALGO" "$ENV" ;;
        esac
    done
done

if [ -n "$DRY_RUN" ]; then
    echo "Dry run: $count jobs would be submitted ($skipped skipped)."
else
    echo "Submitted $count jobs ($skipped skipped)."
fi
