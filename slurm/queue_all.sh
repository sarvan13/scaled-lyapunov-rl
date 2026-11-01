#!/bin/bash
# Queue every experiment (each job is a 10-seed array):
#
#   pendulum, double:  LSAC (every scaling version) + SAC
#   quad:              LPPO and LY (every scaling version) + PPO
#   hopper, walker:    subset LY (every scaling version) + PPO and SAC
#
# "Every scaling version" of a Lyapunov algorithm is:
#   none            unscaled baseline
#   structural      single run (no mu)
#   min_lie         sweep over min_lie_mus below
#   regularization  sweep over reg_mus below
#
# Jobs run the code in $REPO (default ~/projects/def-danielac/sarvan13/stable-lppo/scaled-lyapunov-rl).
# Configs that already have results in $REPO/data are skipped unless FORCE=1.
#
# Usage (run directly on the cluster, not with sbatch):
#   bash slurm/queue_all.sh                     # every environment
#   bash slurm/queue_all.sh hopper walker       # only some environments
#   ALGOS="lppo ppo" bash slurm/queue_all.sh    # only some algorithms (lsac sac lppo ly ly_subset ppo)
#   DRY_RUN=1 bash slurm/queue_all.sh           # print the sbatch commands without submitting
#   FORCE=1 bash slurm/queue_all.sh             # also resubmit configs that already have results

export REPO=${REPO:-$HOME/projects/def-danielac/sarvan13/stable-lppo/scaled-lyapunov-rl}
if [ -d "$REPO" ]; then
    cd "$REPO"
elif [ -n "$DRY_RUN" ]; then
    echo "note: $REPO not found; dry run uses this checkout"
    cd "$(dirname "$0")/.."
else
    echo "REPO not found: $REPO"
    exit 1
fi

ENVS=${@:-pendulum double quad hopper walker}

# Algorithms per environment: scaled Lyapunov algorithms first, then the unscaled baselines
lyap_algos() {
    case "$1" in
        pendulum|double) echo lsac ;;
        quad)            echo lppo ly ;;
        hopper|walker)   echo ly_subset ;;
    esac
}
base_algos() {
    case "$1" in
        pendulum|double) echo sac ;;
        quad)            echo ppo ;;
        hopper|walker)   echo ppo sac ;;
    esac
}

# Sweep values per environment
min_lie_mus() {
    case "$1" in
        pendulum)           echo 0.01 0.1 0.15 0.25 0.5 0.75 ;;
        double)             echo 0.05 0.1 0.15 0.25 0.5 0.75 ;;
        quad|hopper|walker) echo 0.05 0.1 0.5 0.75 ;;
    esac
}
reg_mus() {
    case "$1" in
        pendulum)           echo 0.0001 0.0005 0.001 0.005 0.05 ;;
        double)             echo 0.0001 0.0005 0.001 0.005 0.01 ;;
        quad|hopper|walker) echo 0.0001 0.0005 0.001 0.005 0.01 ;;
    esac
}

wanted() {  # true if ALGOS is unset or contains $1
    [ -z "$ALGOS" ] || [[ " $ALGOS " == *" $1 "* ]]
}

count=0
skipped=0

submit() {  # algo env [method mu]
    local algo=$1 env=$2 method=$3 mu=$4

    # Results folder the python scripts write to (run_tag in algorithms/lyapunov_scaling.py)
    local dir="data/$ENV_ID/clean_$algo"
    case "$method" in
        min_lie|regularization) dir="$dir/${method}_mu$(printf '%g' "$mu")" ;;
        none|structural)        dir="$dir/$method" ;;
    esac
    if [ -z "$FORCE" ] && [ -d "$dir" ]; then
        echo "skip ($dir exists): $algo $env $method $mu"
        skipped=$((skipped + 1))
        return
    fi

    if [ -n "$DRY_RUN" ]; then
        echo sbatch slurm/queue_$algo.sh "$env" $method $mu
    else
        echo -n "$algo $env $method $mu -> "
        sbatch slurm/queue_$algo.sh "$env" $method $mu
    fi
    count=$((count + 1))
}

for ENV in $ENVS; do
    case "$ENV" in
        pendulum|double|quad|hopper|walker) ;;
        *) echo "Unknown environment '$ENV' (use pendulum, double, quad, hopper or walker)"; exit 1 ;;
    esac
    source slurm/envs.sh  # sets ENV_ID

    for ALGO in $(lyap_algos "$ENV"); do
        wanted "$ALGO" || continue
        submit "$ALGO" "$ENV" none 0.0
        submit "$ALGO" "$ENV" structural 0.0
        for MU in $(min_lie_mus "$ENV"); do
            submit "$ALGO" "$ENV" min_lie "$MU"
        done
        for MU in $(reg_mus "$ENV"); do
            submit "$ALGO" "$ENV" regularization "$MU"
        done
    done

    for ALGO in $(base_algos "$ENV"); do
        wanted "$ALGO" || continue
        submit "$ALGO" "$ENV"
    done
done

if [ -n "$DRY_RUN" ]; then
    echo "Dry run: $count jobs would be submitted ($skipped skipped)."
else
    echo "Submitted $count jobs ($skipped skipped)."
fi
