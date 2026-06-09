"""
Headless driver for the big BAL sweep on the FAS cluster.

Runs 7 methods on a 5x5 (tau_r, tau_kappa) grid at T=60, N=15 using
`run_grid_sweep_parallel` with all available cores. All BT methods use
hit-and-run MCMC posteriors (Laplace dropped). The methods isolate two axes:
  - Response granularity: 4-outcome (multiframe), 3-outcome (multiframe_3outcome),
    2-outcome (bt_*).
  - Noise-family knowledge: known logistic vs MoG-fit (multiframe_unknown_family,
    bt_mog).

bt_hitandrun (and its random/lex forced-choice variants) uses fixed_scale=
1/scale_delta so it's apples-to-apples with multiframe — same likelihood at
tau=tau'=0, same posterior approximation, only response granularity differs.

Loads helper functions directly from the notebook, so this stays in sync with
whatever you have in `BALD_bt_vs_multiframe_experiment.ipynb` — no copy-paste.

Run order on the cluster:
    1. sbatch sbatch_test_sweep.sh       # ~1-2 min, validates the pipeline
    2. inspect test_sweep_<ts>.pkl
    3. sbatch sbatch_big_sweep.sh        # ~hours, the real run
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
    """Exec all foundation code cells from the notebook into a fresh namespace.

    Identifies helper cells by content signature so this is robust to the
    notebook being reordered. Skips the smoke/sweep/plot cells (which we
    only want triggered by this script's own main(), not as a side-effect of import).
    """
    nb = json.loads(nb_path.read_text())
    ns = {"__name__": "__main__", "np": np}

    # Signatures that mark a foundation cell. Order matters: imports first,
    # then everything else (each cell's helpers depend on prior ones).
    SIGNATURES = [
        "FEATURE_NAMES = [",                  # cell 1: imports + constants
        "def predict_response_noisy",         # cell 2: Utilize-Indecision model
        "def make_holdout_set",               # cell 3: holdout + log-loss
        "def sample_posterior_hit_and_run",   # cell 4: MCMC samplers
        "def multiframe_bald_score",          # cell 5: BALD
        "def run_multiframe_trial",           # cell 6: Phase-1/2 trial runners
        "def run_bt_trial_laplace_bald",      # cell 7: BT Laplace trial runner
        "def run_bt_trial_hitandrun",         # cell 8: BT hit-and-run trial runner (apples-to-apples MCMC)
        "def run_cell_experiment",            # cell 9: dispatcher (where make_fixed_oracle_set lives)
        "def run_grid_sweep",                 # cell 10: serial grid sweep
        "def sample_posterior_mog",           # cell 11: Phase-3 helpers
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
    taus=[0.0, 0.2, 0.4, 0.6, 0.8],
    tau_primes=[0.0, 0.2, 0.4, 0.6, 0.8],
    T=60,
    N=15,
    noise_type="logistic",           # oracle and all learners use logistic noise.
    scale_delta=0.1,                 # logistic scale s — same parameter BT uses internally.
    scale_r=0.0,
    lambda_x=1.0,
    shape_beta=2.0,                  # only used if noise_type == 'gennorm' — ignored here.
    n_candidates=50,
    n_posterior_samples=200,
    methods=[
        "multiframe",
        "multiframe_3outcome",
        "multiframe_unknown_family",
        "bt_mog",
        "bt_hitandrun",          # apples-to-apples MCMC counterpart to multiframe (logistic, fixed scale)
        "bt_hitandrun_random",   # + random forced-choice
        "bt_hitandrun_lex",      # + lexicographic forced-choice
    ],
    lex_ranking=[1, 4, 2, 3, 0],
    seed=42,
    oracle_seed=20,
    oracle_alpha=0.2,
    learn_scale=True,
)


def main():
    print(f"[{datetime.now().isoformat(timespec='seconds')}] starting", flush=True)
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
    pkl_path = HERE / f"grid_sweep_7methods_{ts}.pkl"
    metadata = {**CONFIG, "elapsed_seconds": elapsed, "n_jobs": n_jobs}
    metadata["description"] = (
        "5x5 (tau_r, tau_kappa) grid with T=60, N=15. 7 methods, all using "
        "hit-and-run MCMC posteriors (BT Laplace dropped): multiframe, "
        "multiframe_3outcome, multiframe_unknown_family, bt_mog, bt_hitandrun, "
        "bt_hitandrun_random, bt_hitandrun_lex. Oracle DGP: LOGISTIC noise. "
        "Methods isolate response-granularity (4 vs 3 vs 2 outcomes) and "
        "noise-family knowledge (multiframe_unknown_family and bt_mog fit "
        "MoG; others assume logistic). bt_hitandrun uses fixed_scale="
        "1/scale_delta to match multiframe's implicit logistic slope."
    )
    with pkl_path.open("wb") as f:
        pickle.dump({"results": results, "metadata": metadata}, f)
    print(f"saved {pkl_path}", flush=True)


if __name__ == "__main__":
    main()
