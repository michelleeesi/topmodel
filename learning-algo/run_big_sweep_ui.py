"""
Headless driver for the UTILIZE-vs-IGNORE comparison on the FAS cluster.

Runs 5 methods on a 5x5 (tau_r, tau_kappa) grid at T=60, N=15 using
`run_grid_sweep_parallel` with all available cores. All methods use hit-and-run
MCMC posteriors (the Laplace path was retired). Comparison axes:

  - Response granularity: 4-outcome (utilize), 3-outcome (utilize_3outcome),
    2-outcome (ignore, ignore_mog).
  - Noise-family knowledge: known logistic (utilize, utilize_3outcome, ignore)
    vs jointly-inferred MoG (utilize_unknown_family, ignore_mog).

ignore uses fixed_scale = 1/scale_delta so it's apples-to-apples with utilize
— same likelihood at tau_r=tau_kappa=0, same posterior approximation, the only
difference is how indecisive responses are handled (dropped vs retained).

Loads helper functions directly from the notebook, so this stays in sync with
whatever is in `BALD_bt_vs_multiframe_experiment.ipynb` — no copy-paste.

Run order on the cluster:
    1. sbatch sbatch_test_sweep_ui.sh    # ~1-2 min, validates the pipeline
    2. inspect test_sweep_ui_<ts>.pkl
    3. sbatch sbatch_big_sweep_ui.sh     # ~hours, the real run
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
    """Exec foundation code cells from the notebook into a fresh namespace,
    identified by content signature so this is robust to reordering."""
    nb = json.loads(nb_path.read_text())
    ns = {"__name__": "__main__", "np": np}

    SIGNATURES = [
        "FEATURE_NAMES = [",                  # cell 1: imports + constants
        "def predict_response_noisy",         # cell 2: response model
        "def make_holdout_set",               # cell 3: holdout + log-loss
        "def sample_posterior_hit_and_run",   # cell 4: hit-and-run MCMC
        "def utilize_bald_score",             # cell 5: 4-outcome BALD
        "def run_utilize_trial",              # cell 6: Utilize trial runners
        "def run_bt_trial_laplace_bald",      # cell 7: BT helpers (legacy, kept for definitions)
        "def run_bt_trial_hitandrun",         # cell 8: BT hit-and-run trial runner
        "def run_cell_experiment",            # cell 9: dispatcher
        "def run_grid_sweep",                 # cell 10: serial grid sweep
        "def sample_posterior_mog",           # cell 11: MoG-noise helpers
        "def run_grid_sweep_parallel",        # cell 12: parallel runner
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


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
CONFIG = dict(
    tau_rs=[0.0, 0.2, 0.4, 0.6, 0.8],
    tau_kappas=[0.0, 0.2, 0.4, 0.6, 0.8],
    T=60,
    N=15,
    noise_type="logistic",
    scale_delta=0.5,                 # logistic scale s — same parameter ignore uses internally.
    scale_r=0.0,
    lambda_x=1.0,
    shape_beta=2.0,                  # only used if noise_type == 'gennorm' — ignored here.
    n_candidates=50,
    n_posterior_samples=200,
    methods=[
        "utilize",
        "utilize_3outcome",
        "utilize_unknown_family",
        "ignore",                    # apples-to-apples MCMC counterpart to utilize
        "ignore_mog",                # noise-agnostic Ignore (joint Gibbs over MoG noise)
    ],
    lex_ranking=[1, 4, 2, 3, 0],     # unused (no force methods) but kept for API compat
    seed=42,
    oracle_seed=20,
    oracle_alpha=0.2,
    learn_scale=True,
)


def main():
    print(f"[{datetime.now().isoformat(timespec='seconds')}] starting (utilize/ignore only)", flush=True)
    print(f"  cwd:        {os.getcwd()}")
    print(f"  notebook:   {NOTEBOOK}")
    print(f"  cpu_count:  {os.cpu_count()}")
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    print(f"  n_jobs:     {n_jobs} (from SLURM_CPUS_PER_TASK or os.cpu_count())")
    print()
    print("CONFIG:")
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")
    print(flush=True)

    ns = load_helpers_from_notebook(NOTEBOOK)
    run = ns["run_grid_sweep_parallel"]

    t0 = time.time()
    results = run(n_jobs=n_jobs, backend="loky", verbose=10, **CONFIG)
    elapsed = time.time() - t0
    print(f"\n[{datetime.now().isoformat(timespec='seconds')}] sweep done in {elapsed/60:.1f} min", flush=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pkl_path = HERE / f"grid_sweep_ui_5methods_{ts}.pkl"
    metadata = {**CONFIG, "elapsed_seconds": elapsed, "n_jobs": n_jobs}
    metadata["description"] = (
        "5x5 (tau_r, tau_kappa) grid with T=60, N=15. 5 methods — utilize, "
        "utilize_3outcome, utilize_unknown_family (MoG noise), ignore, "
        "ignore_mog (MoG noise). All hit-and-run MCMC posteriors. Oracle DGP: "
        "LOGISTIC noise. Methods isolate two axes: (1) response granularity "
        "(4 vs 3 vs 2 outcomes) and (2) noise-family knowledge (logistic known "
        "vs MoG inferred). ignore uses fixed_scale = 1/scale_delta so it's "
        "apples-to-apples with utilize — same likelihood at the (0,0) corner, "
        "same posterior approximation, only indecision handling differs."
    )
    with pkl_path.open("wb") as f:
        pickle.dump({"results": results, "metadata": metadata}, f)
    print(f"saved {pkl_path}", flush=True)


if __name__ == "__main__":
    main()
