"""Plot training curves of the RL / stable-RL algorithms.

Reads data/<env>/<algo>[/<scaling tag>]/seed_*/{returns,steps}.npy as written by the scripts in
algortihms/, and plots the mean episodic reward across seeds (line) with a +/- one standard
deviation band against training steps.

Two modes:

1. Parameter sweeps (--sweep): one plot per environment / algorithm / scaling method, showing every
   mu that was run plus the unscaled run (scaling method "none") as mu = 0. The printed table is
   sorted by final reward, with the run key to paste into --include for the comparison plot.

    python plotting/plot_rewards.py --sweep                                   # every sweep found in data/
    python plotting/plot_rewards.py --sweep --env Pendulum-v1 --algo lsac --method min_lie

2. Comparison (default): the runs given to --include, in that order (which also fixes their colors).
   Run keys are 'algo' or 'algo/tag', where tag is the folder name under data/<env>/<algo>/.

    python plotting/plot_rewards.py --env Pendulum-v1 \\
        --include sac ppo ly lsac/min_lie_mu0.1 lsac/regularization_mu0.0005 lsac/structural \\
        --labels SAC PPO LY "LSAC min Lie" "LSAC regularized" "LSAC structural" --name best

Output goes to plots/sweeps/<env>/<algo>_<method>.{png,pdf} and plots/<env>[_<name>].{png,pdf}.
"""
import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ALGO_NAMES = {
    "clean_sac": "SAC",
    "clean_lsac": "LSAC",
    "clean_ppo": "PPO",
    "clean_lppo": "LPPO",
    "clean_ly": "LY",
}
ALGO_ORDER = ["clean_sac", "clean_ppo", "clean_ly", "clean_lsac", "clean_lppo"]
SWEEP_METHODS = ["min_lie", "regularization"]
METHOD_NAMES = {"min_lie": "min Lie", "regularization": "regularized"}

# Fixed categorical order (validated palette, light mode). Series past 8 reuse hues with dashed lines.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"


def format_steps(x, _):
    if abs(x) >= 1e6:
        return f"{x / 1e6:g}M"
    if abs(x) >= 1e3:
        return f"{x / 1e3:g}k"
    return f"{x:g}"


def normalize_key(key):
    """Accept 'lppo/structural' as well as 'clean_lppo/structural'."""
    algo, sep, tag = key.partition("/")
    if not algo.startswith("clean_"):
        algo = "clean_" + algo
    return algo + sep + tag


def split_tag(tag):
    """'min_lie_mu0.1' -> ('min_lie', 0.1); 'structural' -> ('structural', None)."""
    method, sep, mu = tag.partition("_mu")
    return method, (float(mu) if sep else None)


def tag_label(tag):
    if tag in ("", "none"):
        return ""
    method, mu = split_tag(tag)
    name = METHOD_NAMES.get(method, method)
    return f"{name} μ={mu:g}" if mu is not None else name


def run_label(key):
    algo, _, tag = key.partition("/")
    name = ALGO_NAMES.get(algo, algo)
    extra = tag_label(tag)
    return f"{name} ({extra})" if extra else name


def discover_runs(env):
    """Return {run key: [seed dirs]} where run key is 'algo' or 'algo/tag'."""
    runs = {}
    for algo_dir in sorted(glob.glob(os.path.join(PROJECT_DIR, "data", env, "*"))):
        algo = os.path.basename(algo_dir)
        direct = sorted(glob.glob(os.path.join(algo_dir, "seed_*")))
        if direct:  # no scaling tag (clean_sac, clean_ppo, or older runs)
            runs[algo] = direct
        for tag_dir in sorted(glob.glob(os.path.join(algo_dir, "*"))):
            seeds = sorted(glob.glob(os.path.join(tag_dir, "seed_*")))
            if seeds and not os.path.basename(tag_dir).startswith("seed_"):
                runs[f"{algo}/{os.path.basename(tag_dir)}"] = seeds
    return runs


def run_sort_key(key):
    algo, _, tag = key.partition("/")
    rank = ALGO_ORDER.index(algo) if algo in ALGO_ORDER else len(ALGO_ORDER)
    return (rank, algo, tag not in ("", "none"), tag)


def load_seed(seed_dir, window):
    returns_path = os.path.join(seed_dir, "returns.npy")
    steps_path = os.path.join(seed_dir, "steps.npy")
    if not (os.path.exists(returns_path) and os.path.exists(steps_path)):
        return None
    returns = np.asarray(np.load(returns_path), dtype=float).reshape(-1)
    steps = np.asarray(np.load(steps_path), dtype=float).reshape(-1)
    n = min(len(returns), len(steps))
    if n < window:
        return None
    returns, steps = returns[:n], steps[:n]
    smoothed = np.convolve(returns, np.ones(window) / window, mode="valid")
    return steps[window - 1 :], smoothed


def aggregate(seed_dirs, window, max_steps=None, n_points=500):
    curves = [c for c in (load_seed(d, window) for d in seed_dirs) if c is not None]
    if not curves:
        return None
    # Common step range covered by every seed
    start = max(s[0] for s, _ in curves)
    end = min(s[-1] for s, _ in curves)
    if max_steps is not None:
        end = min(end, max_steps)
    if end <= start:
        return None
    grid = np.linspace(start, end, n_points)
    values = np.stack([np.interp(grid, s, r) for s, r in curves])
    return grid, values.mean(axis=0), values.std(axis=0), len(curves)


def make_plot(runs, entries, title, out_base, args, sort_table=False):
    """entries: list of (run key, legend label), drawn in order with colors in that order."""
    if len(entries) > len(COLORS):
        print(f"note: {len(entries)} series; series past {len(COLORS)} reuse colors with dashed lines")

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    summary = []
    for i, (key, label) in enumerate(entries):
        result = aggregate(runs[key], args.window, args.max_steps)
        if result is None:
            print(f"skipping {key}: no seed has returns.npy/steps.npy with >= {args.window} episodes")
            continue
        grid, mean, std, n_seeds = result
        color, linestyle = COLORS[i % len(COLORS)], "-" if i < len(COLORS) else "--"
        ax.fill_between(grid, mean - std, mean + std, color=color, alpha=0.18, linewidth=0)
        ax.plot(grid, mean, color=color, linestyle=linestyle, linewidth=2, label=label)
        tail = max(1, len(grid) // 10)
        summary.append((key, label, mean[-tail:].mean(), std[-tail:].mean(), grid[-1], n_seeds))

    if not summary:
        plt.close(fig)
        return

    ax.set_title(title, color=TEXT, loc="left")
    ax.set_xlabel("Training steps", color=TEXT_SECONDARY)
    ax.set_ylabel(f"Episodic reward ({args.window}-episode moving average)", color=TEXT_SECONDARY)
    ax.xaxis.set_major_formatter(FuncFormatter(format_steps))
    if args.max_steps is not None:
        ax.set_xlim(right=args.max_steps)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY)
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    for fmt in args.formats:
        fig.savefig(f"{out_base}.{fmt}")
    plt.close(fig)
    print(f"saved {out_base} ({', '.join(args.formats)})")

    # Table view: final performance (mean over the last 10% of the plotted step range)
    if sort_table:
        summary.sort(key=lambda s: s[2], reverse=True)
    width = max(len(s[1]) for s in summary)
    key_width = max(len(s[0]) for s in summary)
    print(f"{title} - final reward (last 10% of steps)" + (", best first" if sort_table else ""))
    for key, label, m, s, last, n_seeds in summary:
        print(f"  {label:<{width}}  {m:10.2f} ± {s:8.2f}   {key:<{key_width}}  ({n_seeds} seeds, up to {last:,.0f} steps)")
    print()


def plot_sweeps(env, args):
    runs = discover_runs(env)
    algos = sorted({k.partition("/")[0] for k in runs}, key=run_sort_key)
    if args.algo:
        algos = [a for a in algos if a in {normalize_key(a2) for a2 in args.algo}]
    methods = args.method or SWEEP_METHODS

    found = False
    for algo in algos:
        for method in methods:
            swept = [k for k in runs if k.startswith(f"{algo}/") and split_tag(k.partition("/")[2])[0] == method]
            swept = [k for k in swept if split_tag(k.partition("/")[2])[1] is not None]
            if not swept:
                continue
            found = True
            swept.sort(key=lambda k: split_tag(k.partition("/")[2])[1])
            entries = [(k, f"μ={split_tag(k.partition('/')[2])[1]:g}") for k in swept]
            if f"{algo}/none" in runs:
                entries.insert(0, (f"{algo}/none", "μ=0 (none)"))
            title = f"{env}: {ALGO_NAMES.get(algo, algo)} {METHOD_NAMES.get(method, method)} sweep"
            out_base = os.path.join(args.out, "sweeps", env, f"{algo.replace('clean_', '')}_{method}")
            make_plot(runs, entries, title, out_base, args, sort_table=True)
    if not found:
        print(f"[{env}] no parameter sweeps found")


def plot_comparison(env, args):
    runs = discover_runs(env)
    if args.include:
        keys = []
        for item in map(normalize_key, args.include):
            matched = [item] if item in runs else sorted((k for k in runs if k.startswith(item + "/")), key=run_sort_key)
            if not matched:
                print(f"[{env}] no runs for '{item}'")
            keys += matched
    else:
        keys = sorted(runs, key=run_sort_key)
    if not keys:
        print(f"[{env}] no runs matched")
        return

    if args.labels and len(args.labels) != len(keys):
        print(f"[{env}] --labels has {len(args.labels)} names but {len(keys)} runs matched; using default labels")
    labels = args.labels if args.labels and len(args.labels) == len(keys) else [run_label(k) for k in keys]

    name = f"{env}_{args.name}" if args.name else env
    make_plot(runs, list(zip(keys, labels)), f"{env}: episodic reward", os.path.join(args.out, name), args)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", nargs="*", help="environment ids to plot (default: every folder in data/)")
    parser.add_argument("--sweep", action="store_true", help="plot parameter sweeps instead of a comparison")
    parser.add_argument("--algo", nargs="*", help="sweep mode: algorithms to plot, e.g. lsac lppo (default: all)")
    parser.add_argument("--method", nargs="*", choices=SWEEP_METHODS, help="sweep mode: methods (default: both)")
    parser.add_argument(
        "--include",
        nargs="*",
        help="comparison mode: runs to show, in order, as 'algo' (all its variants) or 'algo/tag', "
        "e.g. sac lsac/structural lsac/min_lie_mu0.1",
    )
    parser.add_argument("--labels", nargs="*", help="comparison mode: legend names, one per run in --include order")
    parser.add_argument("--name", help="comparison mode: suffix for the output file, plots/<env>_<name>")
    parser.add_argument("--window", type=int, default=50, help="moving-average window in episodes (default 50)")
    parser.add_argument("--max-steps", type=float, help="only plot up to this many training steps, e.g. 2e5")
    parser.add_argument("--out", default=os.path.join(PROJECT_DIR, "plots"), help="output folder (default plots/)")
    parser.add_argument("--formats", nargs="*", default=["png", "pdf"], help="file formats (default png pdf)")
    args = parser.parse_args()

    envs = args.env or sorted(os.path.basename(p) for p in glob.glob(os.path.join(PROJECT_DIR, "data", "*")))
    if not envs:
        print(f"no results found in {os.path.join(PROJECT_DIR, 'data')}")
    for env in envs:
        if args.sweep:
            plot_sweeps(env, args)
        else:
            plot_comparison(env, args)


if __name__ == "__main__":
    main()
