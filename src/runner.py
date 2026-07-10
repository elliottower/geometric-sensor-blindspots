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


def run_single_sweep(axis_name, n_per_class, n_devices, seed, baseline_overrides=None):
    """Run all methods and baselines across one decoherence axis sweep."""
    sweep_data = sweep_single_axis(
        axis_name, n_per_class=n_per_class, seed=seed, n_devices=n_devices,
        baseline_overrides=baseline_overrides,
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


def _load_completed(jsonl_path):
    """Load already-completed (axis, seed) pairs and their results from JSONL."""
    done = set()
    results = []
    if jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                key = (entry["axis"], entry["seed"])
                done.add(key)
                results.extend(entry["results"])
    return done, results


def run_all_sweeps(n_per_class, n_devices, n_seeds, base_seed, axes, output_path,
                   baseline_overrides=None):
    """Run sweeps across all axes with N_SEEDS replications per condition.

    Writes results incrementally as JSONL (one line per axis-seed batch).
    On restart, loads the JSONL and skips already-completed (axis, seed) pairs.
    Final combined JSON is written at the end.
    """
    if axes is None:
        axes = list(DECOHERENCE_AXES.keys())

    seeds = [base_seed + s for s in range(n_seeds)]
    n_methods = len(METHODS) + len(BASELINES)
    total_combos = sum(len(DECOHERENCE_AXES[ax]) for ax in axes)
    total_evals = total_combos * n_methods * n_seeds

    jsonl_path = Path(str(output_path) + "l")
    done, all_results = _load_completed(jsonl_path)
    n_skipped = len(done)

    print(f"[{datetime.now(timezone.utc).isoformat()}] Starting experiment sweep")
    print(f"  Axes: {axes}")
    print(f"  Levels per axis: {[len(DECOHERENCE_AXES[ax]) for ax in axes]}")
    print(f"  Methods: {len(METHODS)} geometric + {len(BASELINES)} baselines = {n_methods}")
    print(f"  Seeds: {n_seeds} (base={base_seed})")
    print(f"  Total method evaluations: {total_evals}")
    print(f"  n_per_class={n_per_class}, n_devices={n_devices}")
    if baseline_overrides:
        print(f"  Baseline overrides: {baseline_overrides}")
    if n_skipped:
        print(f"  Resuming: {n_skipped} axis-seed pairs already done, {len(all_results)} cached results")

    total_pairs = len(axes) * n_seeds
    n_total_errors = 0
    pbar = tqdm(total=total_pairs, initial=n_skipped, desc="Axis x Seed")
    with open(jsonl_path, "a") as jl:
        for axis in axes:
            for seed in seeds:
                if (axis, seed) in done:
                    continue

                t0 = time.time()
                results = run_single_sweep(axis, n_per_class, n_devices, seed,
                                           baseline_overrides=baseline_overrides)
                elapsed = time.time() - t0
                n_errors = sum(1 for r in results if r.get("error"))
                n_total_errors += n_errors
                all_results.extend(results)

                jl.write(json.dumps({"axis": axis, "seed": seed, "results": results}) + "\n")
                jl.flush()

                pbar.set_postfix(axis=axis, seed=seed, errors=n_errors, t=f"{elapsed:.0f}s",
                                 done=len(all_results))
                pbar.update(1)

    pbar.close()

    with open(output_path, "w") as f:
        json.dump({
            "meta": _build_meta(n_per_class, n_devices, n_seeds, base_seed, axes, all_results,
                                baseline_overrides=baseline_overrides),
            "results": all_results,
        }, f)

    print(f"\n[{datetime.now(timezone.utc).isoformat()}] Done. {len(all_results)} results saved to {output_path}")
    print(f"  Incremental log: {jsonl_path} ({n_total_errors} new errors, {n_skipped} resumed)")
    return all_results


def _build_meta(n_per_class, n_devices, n_seeds, base_seed, axes, results,
                baseline_overrides=None):
    config = {
        "n_per_class": n_per_class,
        "n_devices": n_devices,
        "n_seeds": n_seeds,
        "base_seed": base_seed,
        "axes": axes,
    }
    if baseline_overrides:
        config["baseline_overrides"] = baseline_overrides
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "commit_sha": get_commit_sha(),
        "config": config,
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
    parser.add_argument("--delta-t", type=float, default=None,
                        help="Override Delta_T (K) for signal strength sweep")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    baseline_overrides = None
    if args.delta_t is not None:
        baseline_overrides = {"delta_T": args.delta_t}

    run_all_sweeps(
        n_per_class=args.n_per_class,
        n_devices=args.n_devices,
        n_seeds=args.n_seeds,
        base_seed=args.base_seed,
        axes=args.axes,
        output_path=output_path,
        baseline_overrides=baseline_overrides,
    )


if __name__ == "__main__":
    main()
