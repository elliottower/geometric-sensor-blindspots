"""Experiment runner: sweep decoherence axes with replication across seeds.

Each condition (axis x level) is run N_SEEDS times with independent random
seeds. Results are saved per-seed so downstream analysis can compute
bootstrap CIs and test directional hypotheses with proper uncertainty.

Usage:
    uv run --with scipy --with scikit-learn --with tqdm python src/runner.py \
        --output results/sweep_results.json

    # Quick smoke test (1 axis, 3 seeds, small data)
    uv run --with scipy --with scikit-learn --with tqdm python src/runner.py \
        --axes alpha_T2 --n-seeds 3 --n-per-class 30 --n-devices 2 \
        --output results/smoke.json
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nv_diamond import DECOHERENCE_AXES, sweep_single_axis
from methods import METHODS
from baselines import BASELINES


DEFAULT_N_PER_CLASS = 200
DEFAULT_N_DEVICES = 3
DEFAULT_N_SEEDS = 20
DEFAULT_BASE_SEED = 1000


def get_commit_sha():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=Path(__file__).parent.parent
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except FileNotFoundError:
        return "unknown"


def run_single_sweep(axis_name, n_per_class, n_devices, seed):
    """Run all methods and baselines across one decoherence axis sweep."""
    sweep_data = sweep_single_axis(
        axis_name, n_per_class=n_per_class, seed=seed, n_devices=n_devices
    )
    results = []
    for level, X, y, dev in sweep_data:
        for method_id, method_fn in METHODS.items():
            try:
                out = method_fn(X, y, dev)
                score = out.pop("score")
                name = out.pop("name")
                results.append({
                    "axis": axis_name,
                    "level": level,
                    "seed": seed,
                    "method": name,
                    "method_id": method_id,
                    "type": "geometric",
                    "score": float(score) if not isinstance(score, int) else score,
                    "extra": {k: float(v) if isinstance(v, (float, np.floating)) else v
                              for k, v in out.items()},
                })
            except Exception as e:
                results.append({
                    "axis": axis_name,
                    "level": level,
                    "seed": seed,
                    "method": f"method_{method_id}",
                    "method_id": method_id,
                    "type": "geometric",
                    "score": None,
                    "error": str(e),
                })

        for bl_name, bl_fn in BASELINES.items():
            try:
                out = bl_fn(X, y, dev)
                score = out.pop("score")
                name = out.pop("name")
                results.append({
                    "axis": axis_name,
                    "level": level,
                    "seed": seed,
                    "method": name,
                    "method_id": bl_name,
                    "type": "baseline",
                    "score": float(score) if not isinstance(score, int) else score,
                    "extra": {k: float(v) if isinstance(v, (float, np.floating)) else v
                              for k, v in out.items()},
                })
            except Exception as e:
                results.append({
                    "axis": axis_name,
                    "level": level,
                    "seed": seed,
                    "method": bl_name,
                    "method_id": bl_name,
                    "type": "baseline",
                    "score": None,
                    "error": str(e),
                })
    return results


def run_all_sweeps(n_per_class, n_devices, n_seeds, base_seed, axes, output_path):
    """Run sweeps across all axes with N_SEEDS replications per condition.

    Writes results incrementally to output_path (one JSON object per flush)
    so partial results survive interruption.
    """
    if axes is None:
        axes = list(DECOHERENCE_AXES.keys())

    seeds = [base_seed + s for s in range(n_seeds)]
    n_methods = len(METHODS) + len(BASELINES)
    total_combos = sum(len(DECOHERENCE_AXES[ax]) for ax in axes)
    total_evals = total_combos * n_methods * n_seeds

    print(f"[{datetime.now(timezone.utc).isoformat()}] Starting experiment sweep")
    print(f"  Axes: {axes}")
    print(f"  Levels per axis: {[len(DECOHERENCE_AXES[ax]) for ax in axes]}")
    print(f"  Methods: {len(METHODS)} geometric + {len(BASELINES)} baselines = {n_methods}")
    print(f"  Seeds: {n_seeds} (base={base_seed})")
    print(f"  Total method evaluations: {total_evals}")
    print(f"  n_per_class={n_per_class}, n_devices={n_devices}")

    all_results = []
    pbar = tqdm(total=len(axes) * n_seeds, desc="Axis x Seed")
    for axis in axes:
        for seed in seeds:
            t0 = time.time()
            results = run_single_sweep(axis, n_per_class, n_devices, seed)
            elapsed = time.time() - t0
            n_errors = sum(1 for r in results if r.get("error"))
            all_results.extend(results)
            pbar.set_postfix(axis=axis, seed=seed, errors=n_errors, t=f"{elapsed:.0f}s")
            pbar.update(1)

            with open(output_path, "w") as f:
                json.dump({
                    "meta": _build_meta(n_per_class, n_devices, n_seeds, base_seed, axes, all_results),
                    "results": all_results,
                }, f, indent=2)

    pbar.close()
    print(f"\n[{datetime.now(timezone.utc).isoformat()}] Done. {len(all_results)} results saved to {output_path}")
    return all_results


def _build_meta(n_per_class, n_devices, n_seeds, base_seed, axes, results):
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "commit_sha": get_commit_sha(),
        "config": {
            "n_per_class": n_per_class,
            "n_devices": n_devices,
            "n_seeds": n_seeds,
            "base_seed": base_seed,
            "axes": axes,
        },
        "n_results": len(results),
        "n_errors": sum(1 for r in results if r.get("error")),
    }


def main():
    parser = argparse.ArgumentParser(description="Run NV-diamond decoherence sweep experiments")
    parser.add_argument("--output", type=str, default="results/sweep_results.json")
    parser.add_argument("--n-per-class", type=int, default=DEFAULT_N_PER_CLASS)
    parser.add_argument("--n-devices", type=int, default=DEFAULT_N_DEVICES)
    parser.add_argument("--n-seeds", type=int, default=DEFAULT_N_SEEDS)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--axes", nargs="*", default=None,
                        help="Which axes to sweep (default: all)")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    run_all_sweeps(
        n_per_class=args.n_per_class,
        n_devices=args.n_devices,
        n_seeds=args.n_seeds,
        base_seed=args.base_seed,
        axes=args.axes,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
