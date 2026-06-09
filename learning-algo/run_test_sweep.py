"""
Smoke-test driver: same code path as run_big_sweep.py but a tiny config so it
finishes in 2-5 minutes. Validates that the cluster env, notebook loading,
joblib parallelism, and pkl saving all work before you kick off the real run.

Runs all 7 production methods (multiframe + 3 multiframe variants + bt_mog +
3 bt_hitandrun variants) on a single cell so any wiring breakage shows up
immediately. BT Laplace methods are intentionally dropped — bt_hitandrun is
the apples-to-apples MCMC contrast we actually care about.
"""
import json
import os
import pickle
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "BALD_bt_vs_multiframe_experiment.ipynb"
assert NOTEBOOK.exists(), f"notebook not found at {NOTEBOOK}"


def load_helpers_from_notebook(nb_path):
    """Exec all foundation code cells from the notebook into a fresh namespace,
    identified by content signature so this is robust to reordering."""
    nb = json.loads(nb_path.read_text())
    ns = {"__name__": "__main__", "np": np}

    SIGNATURES = [
        "FEATURE_NAMES = [",
        "def predict_response_noisy",
        "def make_holdout_set",
        "def sample_posterior_hit_and_run",
        "def multiframe_bald_score",
        "def run_multiframe_trial",
        "def run_bt_trial_laplace_bald",
        "def run_bt_trial_hitandrun",
        "def run_cell_experiment",
        "def run_grid_sweep",
        "def sample_posterior_mog",
        "def run_grid_sweep_parallel",
    ]

    cell_sources = ["".join(c.get("source", [])) for c in nb["cells"]]
    matched_indices = []
    for sig in SIGNATURES:
        for i, src in enumerate(cell_sources):
            if sig in src and i not in matched_indices:
                matched_indices.append(i)
                exec(compile(src, f"<cell {i}>", "exec"), ns)
                break
        else:
            raise RuntimeError(f"could not find a cell containing {sig!r}")

    return ns


# Tiny config: 1 cell, 4 trials, T=6, low MC budgets. ~2 min on 8 cores.
CONFIG = dict(
    taus=[0.4],
    tau_primes=[0.4],
    T=6,
    N=4,
    noise_type="logistic",
    scale_delta=0.1,
    scale_r=0.0,
    lambda_x=1.0,
    shape_beta=2.0,    # ignored when noise_type='logistic'
    n_candidates=15,
    n_posterior_samples=25,
    methods=[
        "multiframe",
        "multiframe_3outcome",
        "multiframe_unknown_family",
        "bt_mog",
        "bt_hitandrun",
        "bt_hitandrun_random",
        "bt_hitandrun_lex",
    ],
    lex_ranking=[1, 4, 2, 3, 0],
    seed=42,
    oracle_seed=20,
    oracle_alpha=0.2,
    learn_scale=True,
)


def main():
    print(f"[{datetime.now().isoformat(timespec='seconds')}] starting TEST sweep", flush=True)
    print(f"  cwd:        {os.getcwd()}")
    print(f"  notebook:   {NOTEBOOK}")
    print(f"  cpu_count:  {os.cpu_count()}")
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    print(f"  n_jobs:     {n_jobs} (from SLURM_CPUS_PER_TASK or os.cpu_count())")
    print()
    print("CONFIG (TEST — tiny):")
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")
    print(flush=True)

    ns = load_helpers_from_notebook(NOTEBOOK)
    run = ns["run_grid_sweep_parallel"]

    t0 = time.time()
    results = run(n_jobs=n_jobs, backend="loky", verbose=10, **CONFIG)
    elapsed = time.time() - t0
    print(f"\n[{datetime.now().isoformat(timespec='seconds')}] TEST done in {elapsed:.1f}s", flush=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pkl_path = HERE / f"test_sweep_{ts}.pkl"
    metadata = {**CONFIG, "elapsed_seconds": elapsed, "n_jobs": n_jobs,
                "description": (
                    "TEST sweep: tiny config (1 cell, N=4, T=6) to validate the "
                    "cluster pipeline. Runs 7 methods: multiframe, multiframe_3outcome, "
                    "multiframe_unknown_family, bt_mog, bt_hitandrun, "
                    "bt_hitandrun_random, bt_hitandrun_lex. BT Laplace dropped — "
                    "bt_hitandrun is the apples-to-apples MCMC contrast (fixed_scale="
                    "1/scale_delta to match multiframe's implicit logistic slope)."
                )}
    with pkl_path.open("wb") as f:
        pickle.dump({"results": results, "metadata": metadata}, f)
    print(f"saved {pkl_path}", flush=True)

    # Sanity: print final L1 per method to confirm we got real curves
    print("\n--- final-step results (1 cell) ---")
    cell = results[(CONFIG["taus"][0], CONFIG["tau_primes"][0])]
    for m in CONFIG["methods"]:
        cs = cell[m]["mean"][-1]
        l1 = cell[m]["l1_mean"][-1]
        print(f"  {m:<32} cos={cs:.3f}  L1={l1:.3f}")
    print("\nIf the numbers above look reasonable, submit sbatch_big_sweep.sh next.")


if __name__ == "__main__":
    main()
