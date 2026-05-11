"""
Quick visual check: run multiframe / bt_hitandrun / bt_laplace_bald at tau=tau'=0
on a few seeds and plot cosine similarity. If the sketch is correct, multiframe
and bt_hitandrun should sit on top of each other (up to MCMC noise), and
bt_laplace should hover nearby but not exactly overlap.

Loads the notebook helpers (same machinery as run_test_sweep.py) and execs
run_bt_trial_hitandrun.py into that namespace so all the dependencies resolve.
"""
import json
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "BALD_bt_vs_multiframe_experiment.ipynb"
SKETCH = HERE / "run_bt_trial_hitandrun.py"


def load_helpers_from_notebook(nb_path):
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
        "def run_cell_experiment",
        "def run_grid_sweep",
        "def sample_posterior_mog",
        "def run_grid_sweep_parallel",
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


# Config — enough trials for smooth curves
N_TRIALS = 12
T = 20
DIM_LOCAL = 5
TAU, TAU_PRIME = 0.0, 0.0
NOISE_TYPE = "logistic"
SCALE_DELTA = 0.5
SCALE_R = 0.0
LAMBDA_X = 1.0
N_CANDIDATES = 30
N_POSTERIOR_SAMPLES = 150
BASE_SEED = 42
ORACLE_SEED = 20
ORACLE_ALPHA = 0.2


def main():
    print(f"loading notebook helpers...", flush=True)
    ns = load_helpers_from_notebook(NOTEBOOK)
    print(f"loaded. exec'ing sketch...", flush=True)
    exec(compile(SKETCH.read_text(), "<sketch>", "exec"), ns)
    assert "run_bt_trial_hitandrun" in ns, "sketch did not define run_bt_trial_hitandrun"
    print(f"  defined: {[k for k in ns if k.startswith(('run_bt_trial_hitandrun','sample_bt_posterior','compute_bt_log'))]}")

    # Pull required symbols out of the notebook namespace
    make_fixed_oracle_set        = ns["make_fixed_oracle_set"]
    create_noise_fn              = ns["create_noise_fn"]
    make_holdout_set             = ns["make_holdout_set"]
    run_multiframe_trial         = ns["run_multiframe_trial"]
    run_bt_trial_laplace_bald    = ns["run_bt_trial_laplace_bald"]
    run_bt_trial_hitandrun       = ns["run_bt_trial_hitandrun"]
    DIM_NB                       = ns["DIM"]

    print(f"  DIM (from notebook): {DIM_NB}")
    oracles = make_fixed_oracle_set(N_TRIALS, dim=DIM_NB,
                                    oracle_seed=ORACLE_SEED, alpha=ORACLE_ALPHA)
    for i, w in enumerate(oracles):
        print(f"    oracle {i}: {np.round(w, 3)}")

    methods = ["multiframe", "bt_hitandrun", "bt_laplace"]
    cos = {m: [] for m in methods}
    l1  = {m: [] for m in methods}

    t0 = time.time()
    for trial in range(N_TRIALS):
        trial_seed = BASE_SEED + trial
        oracle_w   = oracles[trial]

        # ---- shared per-trial RNGs (mirrors run_cell_experiment) ----
        candidate_rng = np.random.default_rng(trial_seed + 700_000)
        holdout_rng   = np.random.default_rng(trial_seed + 500_000)
        # holdout uses its own oracle_rng (independent stream)
        holdout_noise_fn = create_noise_fn(NOISE_TYPE, SCALE_DELTA, SCALE_R,
                                           np.random.default_rng(trial_seed + 600_000))

        holdout = make_holdout_set(50, oracle_w, holdout_noise_fn,
                                   TAU, TAU_PRIME, LAMBDA_X, holdout_rng, V=None)

        # Each method gets a fresh noise_fn with the SAME seed → identical
        # oracle responses for identical queries.
        def fresh_noise_fn():
            r = np.random.default_rng(trial_seed + 800_000)
            return create_noise_fn(NOISE_TYPE, SCALE_DELTA, SCALE_R, r)

        # FIXED SEED: same trial_seed across all methods (different generator
        # objects so each method's rng stream starts at the same state).
        # Combined with shared candidate_rng + identical noise_fn, this means:
        #   - same candidate batches each step
        #   - same oracle responses for any given query
        #   - MCMC chains see the same proposals + accept/reject draws
        # The only thing left to differ is the likelihood, which is the
        # variable we're isolating.
        SHARED_SEED = trial_seed
        # BT must use the same effective slope multi-frame uses implicitly:
        # multi-frame's logistic noise has scale_delta, giving sigmoid slope
        # 1/scale_delta. Set BT's fixed scale to match.
        FIXED_BT_SCALE = 1.0 / SCALE_DELTA  # = 2.0 here

        for method_index, m in enumerate(methods):
            method_rng = np.random.default_rng(SHARED_SEED)
            cand_rng_per_method = np.random.default_rng(trial_seed + 700_000)

            t1 = time.time()
            if m == "multiframe":
                r = run_multiframe_trial(
                    oracle_weights=oracle_w, noise_type=NOISE_TYPE,
                    scale_delta=SCALE_DELTA, scale_r=SCALE_R,
                    tau=TAU, tau_prime=TAU_PRIME, lambda_x=LAMBDA_X,
                    n_attempts=T, n_candidates=N_CANDIDATES,
                    n_posterior_samples=N_POSTERIOR_SAMPLES,
                    holdout=holdout, V=None, rng=method_rng,
                    candidate_rng=cand_rng_per_method,
                    noise_fn=fresh_noise_fn(),
                )
            elif m == "bt_hitandrun":
                r = run_bt_trial_hitandrun(
                    oracle_weights=oracle_w, noise_type=NOISE_TYPE,
                    scale_delta=SCALE_DELTA, scale_r=SCALE_R,
                    tau=TAU, tau_prime=TAU_PRIME, lambda_x=LAMBDA_X,
                    n_attempts=T, n_candidates=N_CANDIDATES,
                    n_posterior_samples=N_POSTERIOR_SAMPLES,
                    burn_in=150,
                    holdout=holdout, V=None,
                    learn_scale=False, fixed_scale=FIXED_BT_SCALE,
                    rng=method_rng,
                    candidate_rng=cand_rng_per_method,
                    noise_fn=fresh_noise_fn(),
                )
            elif m == "bt_laplace":
                r = run_bt_trial_laplace_bald(
                    oracle_weights=oracle_w, noise_type=NOISE_TYPE,
                    scale_delta=SCALE_DELTA, scale_r=SCALE_R,
                    tau=TAU, tau_prime=TAU_PRIME, lambda_x=LAMBDA_X,
                    n_attempts=T, n_candidates=N_CANDIDATES,
                    n_posterior_samples=N_POSTERIOR_SAMPLES,
                    holdout=holdout, V=None, learn_scale=True,
                    rng=method_rng,
                    candidate_rng=cand_rng_per_method,
                    noise_fn=fresh_noise_fn(),
                )
            cos[m].append(r["cos_sims"])
            l1[m].append(r["l1s"])
            print(f"  trial {trial+1}/{N_TRIALS}  {m:<14}  {time.time()-t1:6.1f}s  "
                  f"final cos={r['cos_sims'][-1]:.3f}", flush=True)

    print(f"\ntotal: {time.time()-t0:.1f}s")

    # ---- plot ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = {"multiframe": "#4477AA",
              "bt_hitandrun": "#CC3333",
              "bt_laplace": "#398C46"}
    labels = {"multiframe": "Utilize-Indecision (MCMC)",
              "bt_hitandrun": "BT (hit-and-run MCMC) — NEW",
              "bt_laplace": "BT (Laplace) — old"}

    for ax, key, title in zip(axes, [cos, l1], ["Cosine similarity", "L1 error"]):
        for m in methods:
            arr = np.array(key[m])
            mean = arr.mean(axis=0)
            sem = arr.std(axis=0) / np.sqrt(N_TRIALS)
            x = np.arange(1, T + 1)
            ax.plot(x, mean, color=colors[m], label=labels[m], linewidth=2)
            ax.fill_between(x, mean - sem, mean + sem, color=colors[m], alpha=0.2)
        ax.set_xlabel("Queries")
        ax.set_ylabel(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle(
        f"Apples-to-apples check: τ=τ'=0, logistic, N={N_TRIALS} trials, T={T}",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()

    out_path = HERE / "bt_hitandrun_check.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
