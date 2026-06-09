"""
Re-run ONLY the (tau_r, tau_kappa) = (0, 0) cell.

Matches run_big_sweep_ui.py's CONFIG exactly except tau_rs/tau_kappas pinned to [0.0].
Saves a small pkl that can be spliced into the full cluster pkl post-hoc on the laptop.

Use this after a notebook fix that only affects the (0, 0) corner (e.g. the
reduction-theorem fast-path) — saves you re-running the other 24 cells.
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
    nb = json.loads(nb_path.read_text())
    ns = {"__name__": "__main__", "np": np}
    SIGNATURES = [
        "FEATURE_NAMES = [",
        "def predict_response_noisy",
        "def make_holdout_set",
        "def sample_posterior_hit_and_run",
        "def utilize_bald_score",
        "def run_utilize_trial",
        "def run_bt_trial_laplace_bald",
        "def run_bt_trial_hitandrun",
        "def run_cell_experiment",
        "def run_grid_sweep",
        "def sample_posterior_mog",
        "def run_grid_sweep_parallel",
        "def sample_posterior_threshold_learning",
    ]
    cell_sources = ["".join(c.get("source", [])) for c in nb["cells"]]
    matched = []
    for sig in SIGNATURES:
        for i, src in enumerate(cell_sources):
            if sig in src and i not in matched:
                matched.append(i)
                exec(compile(src, f"<cell {i}>", "exec"), ns)
                break
        else:
            raise RuntimeError(f"could not find a cell containing {sig!r}")
    return ns


CONFIG = dict(
    tau_rs=[0.0],
    tau_kappas=[0.0],
    T=100,
    N=30,
    noise_type="logistic",
    scale_delta=0.1,
    scale_r=0.0,
    lambda_x=1.0,
    shape_beta=2.0,
    n_candidates=50,
    n_posterior_samples=200,
    methods=[
        "utilize",
        "utilize_3outcome",
        "utilize_unknown_family",
        "utilize_threshold_learning",
        "ignore",
        "ignore_mog",
    ],
    lex_ranking=[1, 4, 2, 3, 0],
    seed=42,
    oracle_seed=20,
    oracle_alpha=0.2,
    learn_scale=True,
)


def main():
    print(f"[{datetime.now().isoformat(timespec='seconds')}] starting corner-only run", flush=True)
    print(f"  cwd:        {os.getcwd()}")
    print(f"  notebook:   {NOTEBOOK}")
    print(f"  cpu_count:  {os.cpu_count()}")
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    print(f"  n_jobs:     {n_jobs} (from SLURM_CPUS_PER_TASK or os.cpu_count())")
    print()
    print("CONFIG (corner only — tau_r=tau_kappa=0):")
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")
    print(flush=True)

    ns = load_helpers_from_notebook(NOTEBOOK)
    run = ns["run_grid_sweep_parallel"]

    t0 = time.time()
    results = run(n_jobs=n_jobs, backend="loky", verbose=10, **CONFIG)
    elapsed = time.time() - t0
    print(f"\n[{datetime.now().isoformat(timespec='seconds')}] done in {elapsed/60:.1f} min", flush=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pkl_path = HERE / f"corner_only_5methods_{ts}.pkl"
    metadata = {**CONFIG, "elapsed_seconds": elapsed, "n_jobs": n_jobs}
    metadata["description"] = (
        "Corner-only re-run of (tau_r, tau_kappa) = (0, 0) at T=100, N=30. "
        "5 methods (utilize, utilize_3outcome, utilize_unknown_family, ignore, "
        "ignore_mog). Intended to splice into a full grid_sweep_ui_5methods_*.pkl "
        "from the big sweep, replacing only the (0, 0) entry. Uses the same "
        "seeds (seed=42, oracle_seed=20, oracle_alpha=0.2) so the result is a "
        "drop-in replacement assuming the rest of the config matches."
    )
    with pkl_path.open("wb") as f:
        pickle.dump({"results": results, "metadata": metadata}, f)
    print(f"saved {pkl_path}", flush=True)

    # Sanity: print final cos + L1 per method
    cell = results[(0.0, 0.0)]
    print("\n--- final-step (0, 0) per method ---")
    for m in CONFIG["methods"]:
        cs = cell[m]["mean"][-1]
        l1 = cell[m]["l1_mean"][-1]
        print(f"  {m:<28} cos={cs:.6f}  L1={l1:.6f}")
    print("\nExpected (with the reduction-corner fast-path):")
    print("  utilize == utilize_3outcome == ignore       (non-MoG band)")
    print("  utilize_unknown_family == ignore_mog        (MoG band)")


if __name__ == "__main__":
    main()
