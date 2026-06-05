"""
indecision_core.py
==================

Clean, focused reimplementation of the pairwise-comparison-with-indecision model
used in Section 3 of the paper. This is a reorganization of the experiments in
``learning-algo/BALD_bt_vs_multiframe_experiment.ipynb``, stripped down to exactly
what the three Section-3 experiment blocks need.

Setting
-------
Perfectly separable linear-score model. A *query* is a pair of feature vectors
``(x_left, x_right)``; the only thing the model cares about is the feature
**difference** ``delta = x_left - x_right``. Given true weights ``omega*`` (on the
simplex) the two latent quantities are

    g = <omega*, delta>          # directional evidence (signed)
    r = <omega*, |delta|>        # total evidence (magnitude of disagreement)

The *threshold response model* adds a noisy margin ``g_tilde = g + eps`` and two
forms of indecision:

    * low-intensity INDIFFERENCE   when  r < tau_r            (noiseless: depends on r only)
    * high-intensity CONFLICT      when  |g_tilde| <= tau_kappa * r

otherwise the respondent answers LEFT (g_tilde > 0) or RIGHT (g_tilde < 0).

When ``tau_r = tau_kappa = 0`` no response is ever indifferent or conflicted and
the model reduces to ordinary logistic Bradley-Terry with slope ``1/noise_scale``
(this is the Block-0 reduction theorem).

Learners
--------
Every learner uses the *same* hit-and-run MCMC sampler on the simplex; they differ
only in their likelihood / outcome alphabet:

    * ``broad``   -- observes the full 4-outcome alphabet {LEFT, RIGHT, INDIFFERENT, CONFLICT}
    * ``broad3``  -- observes 3 outcomes {LEFT, RIGHT, UNKNOWN}  (indiff + conflict collapsed)
    * ``bt``      -- binary logistic Bradley-Terry on decisive labels only; indecisive
                     responses are either skipped or coerced to LEFT/RIGHT by a *forcing rule*

This isolates the response alphabet as the only moving part between methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.special import expit as sigmoid
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FEATURE_NAMES = ["elderlyDep", "lifeYearsGained", "obesity", "weeklyWorkhours", "yearsWaiting"]
DIM = len(FEATURE_NAMES)

# Outcome alphabet indices (canonical order used everywhere).
LEFT, RIGHT, INDIFFERENT, CONFLICT = 0, 1, 2, 3
LABELS4 = ("left", "right", "indifferent", "conflict")
LABEL_IDX = {lab: i for i, lab in enumerate(LABELS4)}


# ---------------------------------------------------------------------------
# Query representation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Query:
    """A pairwise comparison. Stores both options so option-dependent forcing
    rules (e.g. similarity-to-self) remain well defined; ``delta`` is derived."""

    xl: np.ndarray
    xr: np.ndarray

    @property
    def delta(self) -> np.ndarray:
        return self.xl - self.xr


def _draw_unit(n: int, dim: int, rng: np.random.Generator, cov: Optional[np.ndarray] = None) -> np.ndarray:
    """Draw ``n`` option vectors with U[0,1] marginals.

    ``cov=None``: features independent. Otherwise ``cov`` is a (dim, dim) correlation
    matrix and features are drawn via a Gaussian copula -- correlated normals pushed
    through the normal CDF -- so the marginals stay U[0,1] but the features co-vary.
    """
    if cov is None:
        return rng.uniform(0.0, 1.0, size=(n, dim))
    z = rng.multivariate_normal(np.zeros(dim), cov, size=n)
    return norm.cdf(z)


def draw_features(
    n: int, dim: int, rng: np.random.Generator,
    cov: Optional[np.ndarray] = None, scale: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Draw ``n`` option feature vectors with optional cross-feature correlation
    (``cov``) and heterogeneous per-feature scale (``scale``: features lie in [0, scale_j])."""
    u = _draw_unit(n, dim, rng, cov)
    return u if scale is None else u * np.asarray(scale, dtype=float)


def sample_queries(
    n: int,
    dim: int = DIM,
    rng: Optional[np.random.Generator] = None,
    similarity: Optional[float] = None,
    cov: Optional[np.ndarray] = None,
    scale: Optional[np.ndarray] = None,
) -> List[Query]:
    """Sample ``n`` queries.

    ``similarity=None`` (default): both options drawn independently (clearly-distinct pairs).
    ``similarity=sigma``: the right option is the left option plus N(0, sigma) noise (clipped),
    so the two options are *similar* -> small feature differences -> low total evidence r ->
    the respondent is indecisive far more often (the realistic source of indecision).

    ``cov`` / ``scale`` describe the underlying feature space (correlation and per-feature
    scale); ``None`` recovers i.i.d. U[0, 1] features. The similarity coupling is applied in
    the unit (pre-scale) space so it is comparable across scales.
    """
    if rng is None:
        rng = np.random.default_rng()
    xl = _draw_unit(n, dim, rng, cov)
    if similarity is None:
        xr = _draw_unit(n, dim, rng, cov)
    else:
        xr = np.clip(xl + rng.normal(0.0, similarity, size=(n, dim)), 0.0, 1.0)
    if scale is not None:
        s = np.asarray(scale, dtype=float)
        xl, xr = xl * s, xr * s
    return [Query(xl[i], xr[i]) for i in range(n)]


def deltas_of(queries: Sequence[Query]) -> np.ndarray:
    """Stack the feature-difference vectors of a list of queries -> (n, dim)."""
    return np.array([q.delta for q in queries])


# ---------------------------------------------------------------------------
# Evidence & response model (data-generating process)
# ---------------------------------------------------------------------------
def evidence(delta: np.ndarray, omega: np.ndarray) -> Tuple[float, float]:
    """Return (g, r) = (<omega, delta>, <omega, |delta|>)."""
    g = float(omega @ delta)
    r = float(omega @ np.abs(delta))
    return g, r


def _noise_cdf(z: np.ndarray, noise_type: str, scale: float) -> np.ndarray:
    """CDF of the latent-margin noise eps, evaluated at z (centered at 0, given scale)."""
    if noise_type == "logistic":
        return sigmoid(z / scale)
    if noise_type == "normal":
        return norm.cdf(z / scale)
    raise ValueError(f"unknown noise_type {noise_type!r} (use 'logistic' or 'normal')")


def response_probs(
    g: np.ndarray,
    r: np.ndarray,
    tau_r: float,
    tau_kappa: float,
    noise_scale: float,
    noise_type: str = "logistic",
) -> np.ndarray:
    """Vectorized P(y | g, r) over the 4-outcome alphabet.

    ``g`` and ``r`` may be arrays of matching shape; the returned array has a
    trailing axis of size 4: [p_left, p_right, p_indifferent, p_conflict].

    Indifference is *noiseless*: it depends only on whether r < tau_r. Conflict
    and the left/right split depend on the noisy margin g_tilde = g + eps.
    """
    g = np.asarray(g, dtype=float)
    r = np.asarray(r, dtype=float)
    thr = tau_kappa * r
    F_hi = _noise_cdf(thr - g, noise_type, noise_scale)   # P(eps <=  thr - g)
    F_lo = _noise_cdf(-thr - g, noise_type, noise_scale)  # P(eps <= -thr - g)
    p_left = 1.0 - F_hi
    p_right = F_lo
    p_conflict = np.clip(F_hi - F_lo, 0.0, 1.0)
    p_indiff = np.zeros_like(g)
    probs = np.stack([p_left, p_right, p_indiff, p_conflict], axis=-1)

    # Where r < tau_r the response is deterministically indifferent.
    indiff_mask = r < tau_r
    if np.ndim(indiff_mask) == 0:
        if indiff_mask:
            probs = np.array([0.0, 0.0, 1.0, 0.0])
    else:
        probs[indiff_mask] = np.array([0.0, 0.0, 1.0, 0.0])
    return probs


def sample_response(
    query: Query,
    omega: np.ndarray,
    tau_r: float,
    tau_kappa: float,
    noise_scale: float,
    noise_type: str,
    rng: np.random.Generator,
) -> str:
    """Draw one oracle response from the threshold DGP."""
    g, r = evidence(query.delta, omega)
    if r < tau_r:
        return "indifferent"
    if noise_type == "logistic":
        eps = rng.logistic(0.0, noise_scale)
    elif noise_type == "normal":
        eps = rng.normal(0.0, noise_scale)
    else:
        raise ValueError(f"unknown noise_type {noise_type!r}")
    g_tilde = g + eps
    thr = tau_kappa * r
    if abs(g_tilde) <= thr:
        return "conflict"
    return "left" if g_tilde > 0 else "right"


def bt_probs(g: np.ndarray, scale: float) -> np.ndarray:
    """Binary Bradley-Terry probabilities [p_left, p_right] = [sigmoid(scale*g), .]."""
    g = np.asarray(g, dtype=float)
    p_left = sigmoid(scale * g)
    return np.stack([p_left, 1.0 - p_left], axis=-1)


# ---------------------------------------------------------------------------
# Hit-and-run MCMC on the simplex (shared by every learner)
# ---------------------------------------------------------------------------
def hit_and_run_step(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One hit-and-run step on the probability simplex {w : sum w = 1, w >= 0}."""
    dim = len(x)
    d = rng.standard_normal(dim)
    d -= d.mean()
    nrm = np.linalg.norm(d)
    if nrm < 1e-12:
        return x.copy()
    d /= nrm
    t_min, t_max = -np.inf, np.inf
    for j in range(dim):
        if d[j] > 1e-12:
            t_min = max(t_min, -x[j] / d[j])
        elif d[j] < -1e-12:
            t_max = min(t_max, -x[j] / d[j])
    if t_min >= t_max - 1e-12:
        return x.copy()
    t = rng.uniform(t_min, t_max)
    new_x = np.maximum(x + t * d, 0.0)
    return new_x / new_x.sum()


def mcmc_posterior(
    loglik: Callable[[np.ndarray], float],
    dim: int,
    rng: np.random.Generator,
    n_samples: int = 200,
    burn_in: int = 100,
    x0: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Metropolis hit-and-run sampler for a posterior over the simplex.

    ``loglik`` maps a weight vector on the simplex to a scalar log-likelihood
    (the prior is uniform on the simplex, so it drops out of the MH ratio).
    Returns an (n_samples, dim) array of posterior draws.
    """
    omega = np.ones(dim) / dim if x0 is None else x0.copy()
    ll = loglik(omega)
    samples = np.empty((n_samples, dim))
    kept = 0
    for step in range(burn_in + n_samples):
        prop = hit_and_run_step(omega, rng)
        ll_prop = loglik(prop)
        if np.log(rng.random()) < ll_prop - ll:
            omega, ll = prop, ll_prop
        if step >= burn_in:
            samples[kept] = omega
            kept += 1
    return samples


# ---------------------------------------------------------------------------
# Likelihood builders (close over the transcript / decisive data)
# ---------------------------------------------------------------------------
def make_loglik_broad(
    deltas: np.ndarray,
    label_idx: np.ndarray,
    tau_r: float,
    tau_kappa: float,
    noise_scale: float,
    noise_type: str,
    n_outcomes: int = 4,
) -> Callable[[np.ndarray], float]:
    """Log-likelihood for the broad-alphabet learner.

    ``deltas``: (M, dim) feature differences. ``label_idx``: (M,) integer codes.
    With ``n_outcomes == 3`` the observed label codes are {0:left, 1:right,
    2:unknown} and indifferent+conflict probabilities are summed.
    """
    abs_deltas = np.abs(deltas)
    li = label_idx

    def loglik(omega: np.ndarray) -> float:
        g = deltas @ omega
        r = abs_deltas @ omega
        p = response_probs(g, r, tau_r, tau_kappa, noise_scale, noise_type)  # (M, 4)
        if n_outcomes == 3:
            p = np.stack([p[:, LEFT], p[:, RIGHT], p[:, INDIFFERENT] + p[:, CONFLICT]], axis=-1)
        chosen = p[np.arange(len(li)), li]
        return float(np.sum(np.log(np.maximum(chosen, 1e-12))))

    return loglik


def make_loglik_bt(deltas: np.ndarray, ys: np.ndarray, scale: float) -> Callable[[np.ndarray], float]:
    """Bradley-Terry log-likelihood over decisive labels (ys: 1=left, 0=right)."""
    if len(deltas) == 0:
        return lambda omega: 0.0

    def loglik(omega: np.ndarray) -> float:
        g = deltas @ omega
        logits = scale * g
        # numerically stable log-sigmoid
        log_p = np.where(logits >= 0, -np.log1p(np.exp(-logits)), logits - np.log1p(np.exp(logits)))
        log_1mp = np.where(logits >= 0, -logits - np.log1p(np.exp(-logits)), -np.log1p(np.exp(logits)))
        return float(np.sum(ys * log_p + (1.0 - ys) * log_1mp))

    return loglik


# ---------------------------------------------------------------------------
# BALD query selection (shared; takes a probs_fn over the learner's alphabet)
# ---------------------------------------------------------------------------
def _entropy(p: np.ndarray, axis: int = -1) -> np.ndarray:
    p = np.clip(p, 1e-15, 1.0)
    return -np.sum(p * np.log(p), axis=axis)


def select_query_bald(
    candidates: List[Query],
    posterior: np.ndarray,
    probs_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    rng: np.random.Generator,
    max_samples: int = 60,
) -> Query:
    """Pick the candidate maximizing BALD = H[E_omega p] - E_omega H[p].

    ``probs_fn(deltas, omega_samples)`` must return an array of shape
    (n_candidates, n_samples, K) of per-outcome probabilities.
    """
    deltas = deltas_of(candidates)
    s = min(len(posterior), max_samples)
    omega_s = posterior[:s]
    P = probs_fn(deltas, omega_s)            # (C, S, K)
    mean_p = P.mean(axis=1)                  # (C, K)
    H_mean = _entropy(mean_p, axis=-1)       # (C,)
    mean_H = _entropy(P, axis=-1).mean(axis=1)  # (C,)
    bald = H_mean - mean_H
    return candidates[int(np.argmax(bald))]


def _broad_probs_fn(tau_r, tau_kappa, noise_scale, noise_type, n_outcomes):
    def fn(deltas: np.ndarray, omega_s: np.ndarray) -> np.ndarray:
        # deltas: (C, d), omega_s: (S, d)
        g = deltas @ omega_s.T            # (C, S)
        r = np.abs(deltas) @ omega_s.T    # (C, S)
        p = response_probs(g, r, tau_r, tau_kappa, noise_scale, noise_type)  # (C, S, 4)
        if n_outcomes == 3:
            p = np.stack([p[..., LEFT], p[..., RIGHT], p[..., INDIFFERENT] + p[..., CONFLICT]], axis=-1)
        return p
    return fn


def _bt_probs_fn(scale):
    def fn(deltas: np.ndarray, omega_s: np.ndarray) -> np.ndarray:
        g = deltas @ omega_s.T            # (C, S)
        return bt_probs(g, scale)         # (C, S, 2)
    return fn


# ---------------------------------------------------------------------------
# Forcing rules: map an indecisive query -> 'left' / 'right'
# ---------------------------------------------------------------------------
def force_5050(query: Query, rng: np.random.Generator, **_) -> str:
    """Unbiased coin flip (benign)."""
    return "left" if rng.random() < 0.5 else "right"


def force_bt_consistent(query: Query, rng: np.random.Generator, *, omega, scale, **_) -> str:
    """Bradley-Terry-consistent forcing: choose LEFT w.p. sigmoid(scale * g) (benign)."""
    g, _ = evidence(query.delta, omega)
    return "left" if rng.random() < sigmoid(scale * g) else "right"


def force_lex(query: Query, rng: np.random.Generator, *, ranking, **_) -> str:
    """Lexicographic: walk features in ``ranking`` order, follow the first that differs."""
    d = query.delta
    for j in ranking:
        if d[j] > 0:
            return "left"
        if d[j] < 0:
            return "right"
    return "left" if rng.random() < 0.5 else "right"


def force_single_feature(query: Query, rng: np.random.Generator, *, feature, **_) -> str:
    """Always follow one feature's sign (a degenerate lexicographic rule)."""
    d = query.delta[feature]
    if d > 0:
        return "left"
    if d < 0:
        return "right"
    return "left" if rng.random() < 0.5 else "right"


def force_self_similarity(query: Query, rng: np.random.Generator, *, self_vec, **_) -> str:
    """Pick the option whose feature vector is closer to a fixed 'self' reference."""
    dl = np.linalg.norm(query.xl - self_vec)
    dr = np.linalg.norm(query.xr - self_vec)
    if dl < dr:
        return "left"
    if dr < dl:
        return "right"
    return "left" if rng.random() < 0.5 else "right"


def force_gut_weights(query: Query, rng: np.random.Generator, *, omega_bias, **_) -> str:
    """Dual-process forcing: when torn, defer to a *different* ('gut') weight vector
    ``omega_bias`` instead of the considered preference. Choose LEFT iff
    <omega_bias, delta> >= 0.

    This generalizes the single-feature and lexicographic rules (those are the special
    cases omega_bias = e_j or a steeply-decaying ranking). It pulls the learned weights
    toward a blend of the true omega* (from genuine decisive answers) and omega_bias
    (from forced ones), so the distortion need not be confined to a single feature."""
    g_bias = float(np.asarray(omega_bias) @ query.delta)
    if g_bias > 0:
        return "left"
    if g_bias < 0:
        return "right"
    return "left" if rng.random() < 0.5 else "right"


def force_compromise(query: Query, rng: np.random.Generator, *, center: float = 0.5, **_) -> str:
    """Extremeness aversion (compromise effect): when torn, pick the *less extreme*
    option -- the one closer to the center of the feature ranges (``center`` per feature).

    Unlike the feature-keyed rules this is not axis-aligned: it biases the learner toward
    central options regardless of any single feature, so the induced weight distortion is
    structural rather than a single-feature inflation."""
    c = np.full_like(query.xl, float(center))
    dl = np.linalg.norm(query.xl - c)
    dr = np.linalg.norm(query.xr - c)
    if dl < dr:
        return "left"
    if dr < dl:
        return "right"
    return "left" if rng.random() < 0.5 else "right"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def l1_error(omega_hat: np.ndarray, omega_star: np.ndarray) -> float:
    return float(np.sum(np.abs(omega_hat - omega_star)))


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(a @ b / (na * nb))


def weight_distortion(omega_hat: np.ndarray, omega_star: np.ndarray) -> np.ndarray:
    """Signed per-feature distortion omega_hat - omega_star."""
    return np.asarray(omega_hat) - np.asarray(omega_star)


def best_of_n_regret(
    omega_hat: np.ndarray,
    omega_star: np.ndarray,
    rng: np.random.Generator,
    n_slates: int = 500,
    slate_size: int = 5,
    dim: int = DIM,
    cov: Optional[np.ndarray] = None,
    scale: Optional[np.ndarray] = None,
) -> float:
    """Mean best-of-N selection regret.

    For each slate of ``slate_size`` random items (drawn from the same feature model
    via ``cov``/``scale``), the learned rule picks argmax omega_hat . x; regret =
    max(omega_star . x) - omega_star . x_picked.
    """
    items = draw_features(n_slates * slate_size, dim, rng, cov, scale).reshape(n_slates, slate_size, dim)
    true_scores = items @ omega_star            # (n_slates, slate_size)
    hat_scores = items @ omega_hat
    picked = np.argmax(hat_scores, axis=1)       # (n_slates,)
    chosen_true = true_scores[np.arange(n_slates), picked]
    best_true = true_scores.max(axis=1)
    return float(np.mean(best_true - chosen_true))


def pairwise_regret(omega_hat: np.ndarray, omega_star: np.ndarray, deltas: np.ndarray) -> float:
    """Fraction of held-out (strictly decisive) pairs where omega_hat picks the
    wrong winner relative to omega_star."""
    g_star = deltas @ omega_star
    g_hat = deltas @ omega_hat
    decisive = np.abs(g_star) > 1e-9
    if not np.any(decisive):
        return float("nan")
    disagree = np.sign(g_hat[decisive]) != np.sign(g_star[decisive])
    return float(np.mean(disagree))


__all__ = [
    "FEATURE_NAMES", "DIM", "LABELS4", "LABEL_IDX",
    "Query", "sample_queries", "draw_features", "deltas_of",
    "evidence", "response_probs", "sample_response", "bt_probs",
    "hit_and_run_step", "mcmc_posterior",
    "make_loglik_broad", "make_loglik_bt",
    "select_query_bald", "_broad_probs_fn", "_bt_probs_fn",
    "force_5050", "force_bt_consistent", "force_lex", "force_single_feature", "force_self_similarity",
    "force_gut_weights", "force_compromise",
    "l1_error", "cosine_sim", "weight_distortion", "best_of_n_regret", "pairwise_regret",
]
