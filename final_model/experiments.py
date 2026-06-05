"""
experiments.py
==============

Unified trial runners and a small averaging harness on top of ``indecision_core``.

Every learner (broad 4-outcome, broad 3-outcome, binary BT with/without forcing)
shares the same hit-and-run MCMC posterior and BALD acquisition; only the outcome
alphabet / likelihood differs. Trials are driven entirely by integer seeds so that
methods within a trial can share the candidate pool and oracle responses while
keeping independent inference randomness -- this is what makes the Block-0
reduction and the Block-2 efficiency comparison apples-to-apples.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

import indecision_core as ic
from indecision_core import (
    DIM, LABEL_IDX, Query,
    sample_queries, sample_response,
    make_loglik_broad, make_loglik_bt, mcmc_posterior,
    select_query_bald, _broad_probs_fn, _bt_probs_fn,
    l1_error, cosine_sim, weight_distortion, best_of_n_regret, pairwise_regret,
)

# Label-code maps for the broad learners.
_CODE4 = {"left": 0, "right": 1, "indifferent": 2, "conflict": 3}
_CODE3 = {"left": 0, "right": 1, "indifferent": 2, "conflict": 2}  # collapse -> "unknown"


# ---------------------------------------------------------------------------
# Per-trial holdout (shared evaluation material)
# ---------------------------------------------------------------------------
def _holdout_deltas(n, dim, seed, feature_cov=None, feature_scale=None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Distinct (decidable) test pairs, drawn from the same feature model as deployment.
    return ic.deltas_of(sample_queries(n, dim, rng, cov=feature_cov, scale=feature_scale))


def _final_metrics(omega_hat, omega_star, holdout_deltas, regret_seed, dim,
                   feature_cov=None, feature_scale=None) -> Dict:
    rng = np.random.default_rng(regret_seed)
    return {
        "l1_final": l1_error(omega_hat, omega_star),
        "cos_final": cosine_sim(omega_hat, omega_star),
        "distortion": weight_distortion(omega_hat, omega_star),
        "best_of_n_regret": best_of_n_regret(omega_hat, omega_star, rng, dim=dim,
                                             cov=feature_cov, scale=feature_scale),
        "pairwise_regret": pairwise_regret(omega_hat, omega_star, holdout_deltas),
        "omega_hat": np.asarray(omega_hat),
    }


# ---------------------------------------------------------------------------
# Broad-alphabet learner (4- or 3-outcome)
# ---------------------------------------------------------------------------
def run_trial_broad(
    omega_star: np.ndarray,
    *,
    n_outcomes: int,
    tau_r: float,
    tau_kappa: float,
    noise_scale: float,
    noise_type: str = "logistic",
    T: int = 40,
    n_candidates: int = 30,
    n_samples: int = 200,
    burn_in: int = 100,
    n_holdout: int = 400,
    cand_seed: int = 0,
    oracle_seed: int = 0,
    mcmc_seed: int = 0,
    holdout_seed: int = 0,
    regret_seed: int = 0,
    query_sigma: Optional[float] = None,
    feature_cov: Optional[np.ndarray] = None,
    feature_scale: Optional[np.ndarray] = None,
) -> Dict:
    dim = len(omega_star)
    code_map = _CODE4 if n_outcomes == 4 else _CODE3
    cand_rng = np.random.default_rng(cand_seed)
    oracle_rng = np.random.default_rng(oracle_seed)
    mcmc_rng = np.random.default_rng(mcmc_seed)
    holdout_deltas = _holdout_deltas(n_holdout, dim, holdout_seed, feature_cov, feature_scale)

    probs_fn = _broad_probs_fn(tau_r, tau_kappa, noise_scale, noise_type, n_outcomes)

    deltas: List[np.ndarray] = []
    codes: List[int] = []
    posterior = mcmc_rng.dirichlet(np.ones(dim), size=n_samples)
    omega_hat = posterior.mean(axis=0)

    l1s, coss, n_dec = [], [], []
    decisive = 0
    for _ in range(T):
        candidates = sample_queries(n_candidates, dim, cand_rng, similarity=query_sigma,
                                    cov=feature_cov, scale=feature_scale)
        q = select_query_bald(candidates, posterior, probs_fn, mcmc_rng)
        resp = sample_response(q, omega_star, tau_r, tau_kappa, noise_scale, noise_type, oracle_rng)
        deltas.append(q.delta)
        codes.append(code_map[resp])
        if resp in ("left", "right"):
            decisive += 1

        loglik = make_loglik_broad(
            np.array(deltas), np.array(codes), tau_r, tau_kappa,
            noise_scale, noise_type, n_outcomes=n_outcomes,
        )
        posterior = mcmc_posterior(loglik, dim, mcmc_rng, n_samples, burn_in, x0=omega_hat)
        omega_hat = posterior.mean(axis=0)
        l1s.append(l1_error(omega_hat, omega_star))
        coss.append(cosine_sim(omega_hat, omega_star))
        n_dec.append(decisive)

    out = {"l1s": np.array(l1s), "cos_sims": np.array(coss), "n_decisive": np.array(n_dec)}
    out.update(_final_metrics(omega_hat, omega_star, holdout_deltas, regret_seed, dim,
                              feature_cov, feature_scale))
    return out


# ---------------------------------------------------------------------------
# Binary Bradley-Terry learner (skip indecisive, or coerce via a forcing rule)
# ---------------------------------------------------------------------------
def run_trial_bt(
    omega_star: np.ndarray,
    *,
    tau_r: float,
    tau_kappa: float,
    noise_scale: float,
    noise_type: str = "logistic",
    forcing: Optional[Callable] = None,
    forcing_kwargs: Optional[Dict] = None,
    bt_scale: Optional[float] = None,
    T: int = 40,
    n_candidates: int = 30,
    n_samples: int = 200,
    burn_in: int = 100,
    n_holdout: int = 400,
    cand_seed: int = 0,
    oracle_seed: int = 0,
    mcmc_seed: int = 0,
    holdout_seed: int = 0,
    regret_seed: int = 0,
    query_sigma: Optional[float] = None,
    feature_cov: Optional[np.ndarray] = None,
    feature_scale: Optional[np.ndarray] = None,
) -> Dict:
    """``forcing=None`` -> skip indecisive responses (the 'ignore' learner).
    Otherwise indecisive responses are coerced to left/right by ``forcing``."""
    dim = len(omega_star)
    if bt_scale is None:
        bt_scale = 1.0 / noise_scale  # correctly-specified slope under the reduction
    forcing_kwargs = dict(forcing_kwargs or {})
    forcing_kwargs.setdefault("scale", bt_scale)
    forcing_kwargs.setdefault("omega", omega_star)

    cand_rng = np.random.default_rng(cand_seed)
    oracle_rng = np.random.default_rng(oracle_seed)
    mcmc_rng = np.random.default_rng(mcmc_seed)
    holdout_deltas = _holdout_deltas(n_holdout, dim, holdout_seed, feature_cov, feature_scale)
    probs_fn = _bt_probs_fn(bt_scale)

    dvecs: List[np.ndarray] = []
    ys: List[float] = []
    posterior = mcmc_rng.dirichlet(np.ones(dim), size=n_samples)
    omega_hat = posterior.mean(axis=0)

    l1s, coss, n_dec, n_forced_list = [], [], [], []
    n_forced = 0
    for _ in range(T):
        candidates = sample_queries(n_candidates, dim, cand_rng, similarity=query_sigma,
                                    cov=feature_cov, scale=feature_scale)
        q = select_query_bald(candidates, posterior, probs_fn, mcmc_rng)
        raw = sample_response(q, omega_star, tau_r, tau_kappa, noise_scale, noise_type, oracle_rng)

        if raw in ("left", "right"):
            label = raw
        elif forcing is None:
            label = None  # skip
        else:
            label = forcing(q, mcmc_rng, **forcing_kwargs)
            n_forced += 1

        if label is not None:
            dvecs.append(q.delta)
            ys.append(1.0 if label == "left" else 0.0)
            loglik = make_loglik_bt(np.array(dvecs), np.array(ys), bt_scale)
            posterior = mcmc_posterior(loglik, dim, mcmc_rng, n_samples, burn_in, x0=omega_hat)
            omega_hat = posterior.mean(axis=0)

        l1s.append(l1_error(omega_hat, omega_star))
        coss.append(cosine_sim(omega_hat, omega_star))
        n_dec.append(len(dvecs))
        n_forced_list.append(n_forced)

    out = {
        "l1s": np.array(l1s), "cos_sims": np.array(coss),
        "n_decisive": np.array(n_dec), "n_forced": np.array(n_forced_list),
        "bt_scale": bt_scale,
    }
    out.update(_final_metrics(omega_hat, omega_star, holdout_deltas, regret_seed, dim,
                              feature_cov, feature_scale))
    return out


# ---------------------------------------------------------------------------
# Multi-trial driver
# ---------------------------------------------------------------------------
def make_oracles(n: int, dim: int = DIM, seed: int = 2026, alpha: float = 0.3) -> List[np.ndarray]:
    """Fixed set of sparse-Dirichlet oracle weight vectors (alpha<1 -> peaky/harder)."""
    rng = np.random.default_rng(seed)
    return [rng.dirichlet(alpha * np.ones(dim)) for _ in range(n)]


def make_adversarial_oracles(
    n: int, dead_feature: int = 0, dim: int = DIM, seed: int = 2026, alpha: float = 0.3
) -> List[np.ndarray]:
    """Sparse oracles whose true weight on ``dead_feature`` is exactly zero.

    Used with a biased forcing rule that keys on ``dead_feature``: the rule fabricates
    labels driven by a feature the truth does not value at all, so the distortion it
    induces is *consistently* in the wrong direction (no averaging it away across oracles)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        w = rng.dirichlet(alpha * np.ones(dim))
        w[dead_feature] = 0.0
        s = w.sum()
        out.append(w / s if s > 0 else np.ones(dim) / dim)
    return out


def _run_one(spec, omega_star, common):
    if spec["kind"] == "broad":
        return run_trial_broad(omega_star, n_outcomes=spec["n_outcomes"], **common)
    if spec["kind"] == "bt":
        return run_trial_bt(
            omega_star,
            forcing=spec.get("forcing"),
            forcing_kwargs=spec.get("forcing_kwargs"),
            bt_scale=spec.get("bt_scale"),
            **common,
        )
    raise ValueError(f"unknown spec kind {spec['kind']!r}")


def run_method(
    spec: Dict,
    oracles: Sequence[np.ndarray],
    *,
    tau_r: float,
    tau_kappa: float,
    noise_scale: float,
    noise_type: str = "logistic",
    T: int,
    base_seed: int = 1000,
    n_jobs: int = 1,
    **shared_kwargs,
) -> List[Dict]:
    """Run one method spec across all oracles. ``spec`` keys:
       ``kind`` in {'broad','bt'}; plus kind-specific keys
       ('n_outcomes' for broad; 'forcing'/'forcing_kwargs'/'bt_scale' for bt).

    Candidate/oracle/holdout/regret seeds depend only on the trial index, so they
    are shared across methods; inference (MCMC) randomness depends on ``base_seed``.
    This keeps comparisons aligned while preventing seed-rigged overlap.

    ``n_jobs != 1`` parallelizes the (independent) trials with joblib.
    """
    def common_for(trial):
        return dict(
            tau_r=tau_r, tau_kappa=tau_kappa, noise_scale=noise_scale,
            noise_type=noise_type, T=T,
            cand_seed=700_000 + trial, oracle_seed=800_000 + trial,
            holdout_seed=500_000 + trial, regret_seed=600_000 + trial,
            mcmc_seed=base_seed + trial, **shared_kwargs,
        )

    if n_jobs == 1:
        return [_run_one(spec, om, common_for(t)) for t, om in enumerate(oracles)]

    from joblib import Parallel, delayed
    return Parallel(n_jobs=n_jobs)(
        delayed(_run_one)(spec, om, common_for(t)) for t, om in enumerate(oracles)
    )


def aggregate_curve(trials: List[Dict], key: str) -> Dict[str, np.ndarray]:
    """Mean +/- stderr of a per-step curve across trials."""
    curves = np.array([t[key] for t in trials])
    mean = curves.mean(axis=0)
    stderr = curves.std(axis=0) / np.sqrt(len(curves)) if len(curves) > 1 else np.zeros_like(mean)
    return {"mean": mean, "stderr": stderr, "raw": curves}


def aggregate_scalar(trials: List[Dict], key: str) -> Dict[str, float]:
    vals = np.array([t[key] for t in trials])
    return {
        "mean": float(vals.mean()),
        "stderr": float(vals.std() / np.sqrt(len(vals))) if len(vals) > 1 else 0.0,
        "raw": vals,
    }


def aggregate_distortion(trials: List[Dict]) -> Dict[str, np.ndarray]:
    """Mean signed per-feature distortion ``omega_hat - omega_star`` across trials."""
    D = np.array([t["distortion"] for t in trials])  # (n_trials, dim)
    return {"mean": D.mean(axis=0), "stderr": D.std(axis=0) / np.sqrt(len(D))}


def relative_distortion(trials: List[Dict], baseline: List[Dict]) -> Dict[str, np.ndarray]:
    """Mean *forcing-induced* distortion: omega_hat(rule) - omega_hat(baseline),
    paired per trial. Removes the BT-vs-threshold misspecification drift shared by
    every binary learner, isolating the effect of the forcing behavior itself."""
    D = np.array([t["omega_hat"] - b["omega_hat"] for t, b in zip(trials, baseline)])
    return {"mean": D.mean(axis=0), "stderr": D.std(axis=0) / np.sqrt(len(D))}


__all__ = [
    "run_trial_broad", "run_trial_bt",
    "make_oracles", "make_adversarial_oracles", "run_method",
    "aggregate_curve", "aggregate_scalar", "aggregate_distortion", "relative_distortion",
]
