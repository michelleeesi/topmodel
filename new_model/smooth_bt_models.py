"""
smooth_bt_models.py
───────────────────
Three scale-dependent Bradley–Terry models, each fitted by MLE.

Model
─────
  P(y_t = 1) = σ( B(s_t) · d_t )

  d_t = log r[i_t] − log r[j_t]     log-ratio (item i beats j iff d_t > 0)
  s_t = r[i_t] + r[j_t]             total scale of the pair

Three parameterisations of B(s):

  1. LogScale    B(s) = β₀ + β₁ log s
  2. Saturating  B(s) = β₀ + β₁(1 − e^{−s/τ})      τ > 0 fixed or grid-searched
  3. Spline      B(s) = Σₖ βₖ φₖ(log s)              φₖ = cubic B-spline basis on log s

Design matrix: X[t, k] = d_t · ψₖ(s_t)  so  η_t = X[t] @ β = B(s_t) · d_t.

Usage (quick)
─────────────
  from smooth_bt_models import fit_all_models, compare_models, plot_B_comparison
  out = fit_all_models(i_idx, j_idx, y, r)
  compare_models(out)
  plot_B_comparison(out, s_obs=s)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
from scipy.optimize import LinearConstraint, minimize
from scipy.special import expit, logit


# ─────────────────────────────────────────────────────────────────────────────
# Feature computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_features(
    i_idx: np.ndarray,
    j_idx: np.ndarray,
    r: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-trial log-ratio d_t and scale s_t from item ratings.

    Parameters
    ----------
    i_idx : int array, shape (T,)   focal item indices
    j_idx : int array, shape (T,)   comparison item indices
    r     : float array, shape (N,)  item ratings, all strictly positive

    Returns
    -------
    d : float array, shape (T,)   log(r[i]) − log(r[j])
    s : float array, shape (T,)   r[i] + r[j]

    Raises
    ------
    ValueError  if any rating is non-positive or non-finite.
    """
    i_idx = np.asarray(i_idx, dtype=int).ravel()
    j_idx = np.asarray(j_idx, dtype=int).ravel()
    r     = np.asarray(r,     dtype=float).ravel()

    if not np.all(np.isfinite(r)):
        raise ValueError("All ratings r must be finite.")
    if np.any(r <= 0.0):
        raise ValueError("All ratings r must be strictly positive (required for log).")

    r_i = r[i_idx]
    r_j = r[j_idx]
    return np.log(r_i) - np.log(r_j), r_i + r_j


# ─────────────────────────────────────────────────────────────────────────────
# Core likelihood functions (shared across all models)
# ─────────────────────────────────────────────────────────────────────────────

def neg_log_likelihood(
    beta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.0,
) -> float:
    """Stable negative log-likelihood with optional L2 ridge penalty.

    NLL(β) = Σ_t [logaddexp(0, z_t) − y_t z_t]  +  (α/2) ‖β‖²
    where z_t = X[t] @ β.
    """
    z = X @ beta
    nll = float(np.sum(np.logaddexp(0.0, z) - y * z))
    if alpha > 0.0:
        nll += 0.5 * alpha * float(np.dot(beta, beta))
    return nll


def nll_gradient(
    beta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.0,
) -> np.ndarray:
    """Gradient: X^T(σ(Xβ) − y) + α β."""
    p = expit(X @ beta)
    g = X.T @ (p - y)
    if alpha > 0.0:
        g = g + alpha * beta
    return g


def nll_hessian(
    beta: np.ndarray,
    X: np.ndarray,
    alpha: float = 0.0,
) -> np.ndarray:
    """Hessian: X^T diag(p(1−p)) X + α I."""
    p = expit(X @ beta)
    w = p * (1.0 - p)
    H = (X * w[:, None]).T @ X
    if alpha > 0.0:
        H = H + alpha * np.eye(len(beta))
    return H


# ─────────────────────────────────────────────────────────────────────────────
# Internal: B-spline basis with exact derivative support
# ─────────────────────────────────────────────────────────────────────────────

class _BSplineBasis:
    """Clamped B-spline basis with quantile-spaced interior knots.

    Each basis function is a scipy.interpolate.BSpline object, so derivatives
    are exact (not finite-differenced).

    Parameters
    ----------
    n_df   : int  Number of basis functions.  Must satisfy n_df ≥ degree + 1.
    degree : int  Polynomial degree (default 3 = cubic).
    """

    def __init__(self, n_df: int = 5, degree: int = 3) -> None:
        if n_df < degree + 1:
            raise ValueError(f"n_df={n_df} must be ≥ degree+1={degree + 1}.")
        self.n_df    = n_df
        self.degree  = degree
        self._splines: Optional[List[BSpline]] = None
        self._x_min:   Optional[float] = None
        self._x_max:   Optional[float] = None

    def fit(self, x: np.ndarray) -> "_BSplineBasis":
        """Place knots at quantiles of x and build all basis functions.

        Parameters
        ----------
        x : 1-D array of observed covariate values (e.g. log s).
        """
        x   = np.asarray(x, dtype=float).ravel()
        d   = self.degree

        # For a clamped (d+1)-repeated-boundary B-spline:
        #   n_basis = n_interior + d + 1
        #   ⟹ n_interior = n_df − d − 1  (≥ 0 by constructor check)
        n_interior = self.n_df - d - 1
        x_min, x_max = float(x.min()), float(x.max())

        if n_interior > 0:
            pcts     = np.linspace(0.0, 100.0, n_interior + 2)[1:-1]
            interior = np.percentile(x, pcts)
        else:
            interior = np.array([], dtype=float)

        # Clamped knot sequence: (d+1) repeats at each boundary
        knots = np.concatenate([
            np.repeat(x_min, d + 1),
            interior,
            np.repeat(x_max, d + 1),
        ])

        n_basis = len(knots) - d - 1
        assert n_basis == self.n_df

        # Build one BSpline per basis function (coefficient = unit vector)
        self._splines = []
        for i in range(n_basis):
            c    = np.zeros(n_basis)
            c[i] = 1.0
            self._splines.append(BSpline(knots, c, d))

        self._x_min = x_min
        self._x_max = x_max
        return self

    def _require_fit(self) -> None:
        if self._splines is None:
            raise RuntimeError("Call .fit(x) before using the basis.")

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Evaluate all basis functions at x.  Returns shape (len(x), n_df)."""
        self._require_fit()
        x = np.asarray(x, dtype=float).ravel()
        return np.column_stack([spl(x) for spl in self._splines])

    def derivative(self, x: np.ndarray, nu: int = 1) -> np.ndarray:
        """Evaluate ν-th derivative of every basis function at x.
        Returns shape (len(x), n_df).
        """
        self._require_fit()
        x = np.asarray(x, dtype=float).ravel()
        return np.column_stack([spl.derivative(nu=nu)(x) for spl in self._splines])


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelFitResult:
    """Return value from every model's .fit() method.

    Attributes
    ----------
    model_name           : human-readable string identifying the model.
    beta_hat             : MLE estimates, shape (K,).
    standard_errors      : √diag(I(β̂)⁻¹), shape (K,) or None if unavailable.
    fitted_probabilities : σ(B(s_t)·d_t) at β̂, shape (T,).
    log_likelihood       : unpenalised log-likelihood at β̂.
    aic                  : 2K − 2·log-lik  (unpenalised).
    bic                  : K·log T − 2·log-lik  (unpenalised).
    diagnostics          : dict with optimizer state and numerical checks.
                           Always includes keys:
                           'success', 'message', 'grad_norm',
                           'X_rank', 'X_cond', 'H_cond',
                           'method', 'constrained', 'alpha_ridge'.
    """

    model_name:           str
    beta_hat:             np.ndarray
    standard_errors:      Optional[np.ndarray]
    fitted_probabilities: np.ndarray
    log_likelihood:       float
    aic:                  float
    bic:                  float
    diagnostics:          Dict

    @property
    def n_params(self) -> int:
        return len(self.beta_hat)

    def summary(self) -> str:
        """One-block text summary."""
        se   = self.standard_errors
        rows = [
            f"── {self.model_name} ──",
            f"  K={self.n_params}  "
            f"log-lik={self.log_likelihood:.4f}  "
            f"AIC={self.aic:.4f}  BIC={self.bic:.4f}",
        ]
        for k, b in enumerate(self.beta_hat):
            s = f"  β[{k}] = {b:+.6f}"
            if se is not None and np.isfinite(se[k]):
                s += f"  (SE={se[k]:.5f})"
            rows.append(s)
        rows.append(
            f"  converged={self.diagnostics['success']}  "
            f"|grad|={self.diagnostics['grad_norm']:.2e}  "
            f"method={self.diagnostics['method']}"
        )
        return "\n".join(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Internal: common fitting engine
# ─────────────────────────────────────────────────────────────────────────────

def _fit_beta(
    X: np.ndarray,
    y: np.ndarray,
    *,
    beta_init:      Optional[np.ndarray] = None,
    alpha_ridge:    float = 0.0,
    use_hessian:    bool  = True,
    constrained:    bool  = False,
    A_pos:          Optional[np.ndarray] = None,
    A_mono:         Optional[np.ndarray] = None,
    positivity_eps: float = 1e-8,
    maxiter:        int   = 3000,
    warn_cond:      float = 1e12,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict]:
    """Internal: MLE for logistic regression on design matrix X.

    Returns (beta_hat, standard_errors_or_None, diagnostics_dict).
    diagnostics always contains 'p_hat' (fitted probs) and 'log_lik'
    (unpenalised log-likelihood).

    Shape constraints (applied only when constrained=True):
        A_pos  @ β ≥ positivity_eps    (positivity on a grid)
        A_mono @ β ≤ 0                 (monotonicity on a grid)
    Both are LinearConstraint objects; only non-None ones are added.
    """
    T, K = X.shape
    y    = np.asarray(y, dtype=float).ravel()

    # ── Numerical health checks ───────────────────────────────────────────────
    rank = int(np.linalg.matrix_rank(X))
    if rank < K:
        warnings.warn(
            f"Design matrix rank-deficient (rank={rank} < K={K}); "
            "estimates may not be unique.", RuntimeWarning,
        )
    sv   = np.linalg.svd(X, full_matrices=False, compute_uv=False)
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0.0 else float("inf")
    if cond > warn_cond:
        warnings.warn(
            f"Design matrix ill-conditioned (cond ≈ {cond:.2e}).", RuntimeWarning,
        )

    # ── Separation check ──────────────────────────────────────────────────────
    if (y == 1).any() and (y == 0).any():
        for k in range(K):
            col = X[:, k]
            if np.all(col[y == 1] > 0) and np.all(col[y == 0] < 0):
                warnings.warn(
                    f"Column {k} may cause complete separation.", RuntimeWarning,
                )

    # ── Initial point ─────────────────────────────────────────────────────────
    b0 = (np.zeros(K, dtype=float) if beta_init is None
          else np.asarray(beta_init, dtype=float).ravel())
    if b0.shape != (K,):
        raise ValueError(f"beta_init must have shape ({K},); got {b0.shape}.")

    fun  = lambda b: neg_log_likelihood(b, X, y, alpha_ridge)
    jac  = lambda b: nll_gradient(b, X, y, alpha_ridge)
    hess = lambda b: nll_hessian(b, X, alpha_ridge)

    # ── Build linear constraints ───────────────────────────────────────────────
    lc: List = []
    if constrained:
        rows_A, rows_lb, rows_ub = [], [], []
        if A_pos is not None and len(A_pos) > 0:
            rows_A.append(A_pos)
            rows_lb.append(np.full(len(A_pos), positivity_eps))
            rows_ub.append(np.full(len(A_pos), np.inf))
        if A_mono is not None and len(A_mono) > 0:
            rows_A.append(A_mono)
            rows_lb.append(np.full(len(A_mono), -np.inf))
            rows_ub.append(np.zeros(len(A_mono)))
        if rows_A:
            A  = np.vstack(rows_A)
            lb = np.concatenate(rows_lb)
            ub = np.concatenate(rows_ub)
            lc = [LinearConstraint(A, lb, ub)]

    # ── Optimise ──────────────────────────────────────────────────────────────
    # trust-constr when we have explicit constraints (it accepts the exact Hessian).
    # L-BFGS-B otherwise (faster, handles bounds, avoids precision-loss warnings).
    method = "trust-constr" if lc else "L-BFGS-B"
    opts   = (
        {"maxiter": maxiter, "verbose": 0}
        if method == "trust-constr"
        else {"maxiter": maxiter, "ftol": 1e-15, "gtol": 1e-8}
    )

    res = minimize(
        fun, b0,
        method=method,
        jac=jac,
        hess=(hess if method == "trust-constr" else None),
        constraints=(lc if lc else ()),
        options=opts,
    )

    if not res.success:
        warnings.warn(f"Optimiser did not converge: {res.message}", RuntimeWarning)

    beta_hat = np.asarray(res.x, dtype=float)

    if np.any(np.abs(beta_hat) > 50.0):
        warnings.warn("Some |β̂| > 50 — near-separation likely.", RuntimeWarning)

    p_hat  = expit(X @ beta_hat)
    n_ext  = int(np.sum((p_hat < 1e-8) | (p_hat > 1.0 - 1e-8)))
    if n_ext > 0:
        warnings.warn(
            f"Extreme fitted probabilities for {n_ext}/{T} trials.", RuntimeWarning,
        )

    # Unpenalised log-likelihood (used for AIC/BIC)
    ll_unpen = -neg_log_likelihood(beta_hat, X, y, alpha=0.0)

    # ── Standard errors from observed Fisher information ──────────────────────
    se:     Optional[np.ndarray] = None
    H_cond: Optional[float]      = None
    if use_hessian:
        try:
            H      = nll_hessian(beta_hat, X, alpha_ridge)
            sv_H   = np.linalg.svd(H, compute_uv=False)
            H_cond = float(sv_H[0] / sv_H[-1]) if sv_H[-1] > 0 else float("inf")
            # Use pseudo-inverse when Hessian is near-singular
            cov = (np.linalg.pinv(H) if H_cond > 1e14
                   else np.linalg.inv(H))
            var = np.diag(cov)
            var = np.where(var >= 0, var, np.nan)
            se  = np.sqrt(var)
        except LinAlgError:
            warnings.warn("Hessian inversion failed.", RuntimeWarning)

    grad_norm = float(np.linalg.norm(nll_gradient(beta_hat, X, y, alpha_ridge)))

    diag: Dict = {
        "success":     bool(res.success),
        "message":     str(res.message),
        "n_iter":      int(res.nit) if hasattr(res, "nit") and res.nit is not None else None,
        "fun":         float(res.fun),
        "log_lik":     float(ll_unpen),
        "grad_norm":   grad_norm,
        "X_rank":      rank,
        "X_cond":      float(cond),
        "H_cond":      H_cond,
        "method":      method,
        "constrained": bool(constrained),
        "alpha_ridge": float(alpha_ridge),
        "p_hat":       p_hat,
        "optimizer_result": res,
    }
    return beta_hat, se, diag


def _make_result(
    model_name: str,
    beta_hat:   np.ndarray,
    se:         Optional[np.ndarray],
    diag:       Dict,
    T:          int,
) -> ModelFitResult:
    """Wrap _fit_beta output into a ModelFitResult."""
    K   = len(beta_hat)
    ll  = diag["log_lik"]
    return ModelFitResult(
        model_name           = model_name,
        beta_hat             = beta_hat,
        standard_errors      = se,
        fitted_probabilities = diag["p_hat"],
        log_likelihood       = ll,
        aic                  = 2.0 * K - 2.0 * ll,
        bic                  = K * np.log(T) - 2.0 * ll,
        diagnostics          = diag,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Model 1: Log-scale
# ─────────────────────────────────────────────────────────────────────────────

class LogScaleModel:
    """B(s) = β₀ + β₁ log s.

    Design matrix (2 columns):
        X[t, 0] = d_t
        X[t, 1] = log(s_t) · d_t

    Derivative:
        B'(s) = β₁ / s

    Shape constraints (optional, linear in β):
        Positivity on grid:   [1, log s_m] @ β ≥ ε   for each s_m
        Monotonicity B'≤0:    β₁ ≤ 0
    """

    name = "LogScale"

    # ── Design matrix ─────────────────────────────────────────────────────────
    @staticmethod
    def build_X(d: np.ndarray, s: np.ndarray) -> np.ndarray:
        """X = [d, log(s)·d],  shape (T, 2)."""
        d = np.asarray(d, float).ravel()
        s = np.asarray(s, float).ravel()
        if np.any(s <= 0):
            raise ValueError("s must be positive (needed for log s).")
        return np.column_stack([d, np.log(s) * d])

    # ── B(s) and derivative ───────────────────────────────────────────────────
    @staticmethod
    def B(s: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """B(s) = β₀ + β₁ log s."""
        return beta[0] + beta[1] * np.log(np.asarray(s, float))

    @staticmethod
    def dB_ds(s: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """B'(s) = β₁ / s."""
        return beta[1] / np.asarray(s, float)

    # ── Constraint matrices ───────────────────────────────────────────────────
    @staticmethod
    def constraint_matrices(
        s_grid: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (A_pos, A_mono) for linear constraints on β.

        A_pos @ β ≥ ε   ⟺   β₀ + β₁ log(s_m) ≥ ε   (positivity)
        A_mono @ β ≤ 0  ⟺   β₁ ≤ 0                   (monotonicity)
        """
        log_s  = np.log(np.asarray(s_grid, float).ravel())
        A_pos  = np.column_stack([np.ones_like(log_s), log_s])
        A_mono = np.array([[0.0, 1.0]])
        return A_pos, A_mono

    # ── Fit ───────────────────────────────────────────────────────────────────
    def fit(
        self,
        d: np.ndarray,
        s: np.ndarray,
        y: np.ndarray,
        *,
        beta_init:      Optional[np.ndarray] = None,
        use_hessian:    bool  = True,
        constrained:    bool  = False,
        s_grid:         Optional[np.ndarray] = None,
        positivity_eps: float = 1e-8,
        maxiter:        int   = 3000,
    ) -> ModelFitResult:
        """Fit B(s) = β₀ + β₁ log s by MLE.

        Parameters
        ----------
        d, s         : features from compute_features().
        y            : binary outcomes in {0, 1}.
        constrained  : enforce B(s) > 0  and  B'(s) ≤ 0  on s_grid.
        s_grid       : required when constrained=True.
        """
        y = np.asarray(y, float).ravel()
        T = len(y)
        X = self.build_X(d, s)

        A_pos, A_mono = None, None
        if constrained:
            if s_grid is None:
                raise ValueError("s_grid is required when constrained=True.")
            A_pos, A_mono = self.constraint_matrices(s_grid)

        beta_hat, se, diag = _fit_beta(
            X, y,
            beta_init=beta_init, use_hessian=use_hessian,
            constrained=constrained, A_pos=A_pos, A_mono=A_mono,
            positivity_eps=positivity_eps, maxiter=maxiter,
        )
        return _make_result(self.name, beta_hat, se, diag, T)


# ─────────────────────────────────────────────────────────────────────────────
# Model 2: Saturating early-drop
# ─────────────────────────────────────────────────────────────────────────────

class SaturatingModel:
    """B(s) = β₀ + β₁(1 − e^{−s/τ}),  τ > 0.

    Design matrix (2 columns):
        X[t, 0] = d_t
        X[t, 1] = (1 − e^{−s_t/τ}) · d_t

    Derivative:
        B'(s) = (β₁/τ) e^{−s/τ}

    Shape constraints (for fixed τ; linear in β):
        Positivity on grid:  [1, 1 − e^{−s_m/τ}] @ β ≥ ε
        Monotonicity B'≤0:   β₁ ≤ 0

    τ selection
    ───────────
    • Pass tau=<float>  for a fixed τ.
    • Pass tau_grid=<array>  to grid-search τ by AIC/BIC/log-lik.
    • If neither is given, fit_all_models() builds a default geometric grid.
    """

    name = "Saturating"

    def __init__(
        self,
        tau:      Optional[float]      = None,
        tau_grid: Optional[np.ndarray] = None,
    ) -> None:
        if tau is not None and tau <= 0:
            raise ValueError("tau must be positive.")
        self.tau      = tau
        self.tau_grid = tau_grid

    # ── Design matrix ─────────────────────────────────────────────────────────
    @staticmethod
    def build_X(d: np.ndarray, s: np.ndarray, tau: float) -> np.ndarray:
        """X = [d, (1 − e^{−s/τ})·d],  shape (T, 2)."""
        d   = np.asarray(d, float).ravel()
        s   = np.asarray(s, float).ravel()
        phi = 1.0 - np.exp(-s / tau)
        return np.column_stack([d, phi * d])

    # ── B(s) and derivative ───────────────────────────────────────────────────
    @staticmethod
    def B(s: np.ndarray, beta: np.ndarray, tau: float) -> np.ndarray:
        """B(s) = β₀ + β₁(1 − e^{−s/τ})."""
        s    = np.asarray(s, float)
        beta = np.asarray(beta, float).ravel()
        return beta[0] + beta[1] * (1.0 - np.exp(-s / tau))

    @staticmethod
    def dB_ds(s: np.ndarray, beta: np.ndarray, tau: float) -> np.ndarray:
        """B'(s) = (β₁/τ) e^{−s/τ}."""
        s    = np.asarray(s, float)
        beta = np.asarray(beta, float).ravel()
        return (beta[1] / tau) * np.exp(-s / tau)

    # ── Constraint matrices ───────────────────────────────────────────────────
    @staticmethod
    def constraint_matrices(
        s_grid: np.ndarray,
        tau:    float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (A_pos, A_mono) for the given τ."""
        s_grid = np.asarray(s_grid, float).ravel()
        phi    = 1.0 - np.exp(-s_grid / tau)
        A_pos  = np.column_stack([np.ones_like(phi), phi])
        A_mono = np.array([[0.0, 1.0]])
        return A_pos, A_mono

    # ── Single-τ fit ──────────────────────────────────────────────────────────
    def _fit_fixed_tau(
        self,
        d: np.ndarray,
        s: np.ndarray,
        y: np.ndarray,
        tau: float,
        *,
        beta_init:      Optional[np.ndarray] = None,
        use_hessian:    bool  = True,
        constrained:    bool  = False,
        s_grid:         Optional[np.ndarray] = None,
        positivity_eps: float = 1e-8,
        maxiter:        int   = 3000,
    ) -> ModelFitResult:
        T = len(y)
        X = self.build_X(d, s, tau)

        A_pos, A_mono = None, None
        if constrained:
            if s_grid is None:
                raise ValueError("s_grid is required when constrained=True.")
            A_pos, A_mono = self.constraint_matrices(s_grid, tau)

        beta_hat, se, diag = _fit_beta(
            X, y,
            beta_init=beta_init, use_hessian=use_hessian,
            constrained=constrained, A_pos=A_pos, A_mono=A_mono,
            positivity_eps=positivity_eps, maxiter=maxiter,
        )
        diag = dict(diag, tau=float(tau))   # store τ for later B(s) evaluation
        return _make_result(f"Saturating(τ={tau:.3g})", beta_hat, se, diag, T)

    # ── Main fit (grid-searches τ if needed) ──────────────────────────────────
    def fit(
        self,
        d: np.ndarray,
        s: np.ndarray,
        y: np.ndarray,
        *,
        beta_init:      Optional[np.ndarray] = None,
        use_hessian:    bool  = True,
        constrained:    bool  = False,
        s_grid:         Optional[np.ndarray] = None,
        positivity_eps: float = 1e-8,
        maxiter:        int   = 3000,
        selection:      str   = "aic",
    ) -> ModelFitResult:
        """Fit B(s) = β₀ + β₁(1 − e^{−s/τ}) by MLE.

        If self.tau is set, uses that fixed τ.
        If self.tau_grid is set, grid-searches τ and picks by `selection`.

        Parameters
        ----------
        selection : criterion for τ selection: 'aic', 'bic', or 'loglik'.
        """
        d = np.asarray(d, float).ravel()
        s = np.asarray(s, float).ravel()
        y = np.asarray(y, float).ravel()

        if self.tau is not None:
            return self._fit_fixed_tau(
                d, s, y, self.tau,
                beta_init=beta_init, use_hessian=use_hessian,
                constrained=constrained, s_grid=s_grid,
                positivity_eps=positivity_eps, maxiter=maxiter,
            )

        if self.tau_grid is None:
            raise ValueError("Either tau or tau_grid must be set.")

        tau_arr = np.asarray(self.tau_grid, float).ravel()
        if np.any(tau_arr <= 0):
            raise ValueError("All tau_grid values must be positive.")

        results: List[ModelFitResult] = []
        for tau in tau_arr:
            try:
                r = self._fit_fixed_tau(
                    d, s, y, tau,
                    beta_init=beta_init, use_hessian=False,   # skip SEs in grid pass
                    constrained=constrained, s_grid=s_grid,
                    positivity_eps=positivity_eps, maxiter=maxiter,
                )
                results.append(r)
            except Exception as exc:
                warnings.warn(f"τ={tau:.3g} failed: {exc}", RuntimeWarning)

        if not results:
            raise RuntimeError("All τ values in tau_grid failed to converge.")

        if selection == "aic":
            scores   = [r.aic for r in results]
            best_idx = int(np.argmin(scores))
        elif selection == "bic":
            scores   = [r.bic for r in results]
            best_idx = int(np.argmin(scores))
        elif selection == "loglik":
            scores   = [r.log_likelihood for r in results]
            best_idx = int(np.argmax(scores))
        else:
            raise ValueError("selection must be 'aic', 'bic', or 'loglik'.")

        best_tau = float(tau_arr[best_idx])
        tau_score_pairs = dict(zip(np.round(tau_arr, 4),
                                   [round(sc, 3) for sc in scores]))
        print(f"[SaturatingModel] Grid search: best τ={best_tau:.4g}  "
              f"({selection}={round(scores[best_idx], 3)})  "
              f"all: {tau_score_pairs}")

        # Refit the best τ with Hessian for SEs
        return self._fit_fixed_tau(
            d, s, y, best_tau,
            beta_init=beta_init, use_hessian=use_hessian,
            constrained=constrained, s_grid=s_grid,
            positivity_eps=positivity_eps, maxiter=maxiter,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Model 3: Spline on log s
# ─────────────────────────────────────────────────────────────────────────────

class SplineModel:
    """B(s) = Σₖ βₖ φₖ(log s),  φₖ = clamped cubic B-spline on log s.

    Design matrix (n_df columns):
        X[t, k] = d_t · φₖ(log s_t)

    Derivative (chain rule: d/ds = (1/s) d/d(log s)):
        B'(s) = (1/s) · Σₖ βₖ φₖ'(log s)

    Shape constraints (linear in β, applied on a grid):
        Positivity:    Φ(log s_m) @ β ≥ ε
                       where Φ[m, k] = φₖ(log s_m)
        Monotonicity:  Φ'(log s_m) @ β ≤ 0
                       where Φ'[m, k] = φₖ'(log s_m)
                       (dividing by s_m > 0 does not change the sign constraint)

    Parameters
    ----------
    n_df        : number of B-spline basis functions.  More → more flexible.
    degree      : polynomial degree (default 3 = cubic).
    alpha_ridge : L2 ridge penalty strength.  Helps numerical stability for
                  large n_df.  AIC/BIC are computed with the unpenalised LL.
    """

    name = "Spline"

    def __init__(
        self,
        n_df:        int   = 5,
        degree:      int   = 3,
        alpha_ridge: float = 0.0,
    ) -> None:
        self.n_df        = n_df
        self.degree      = degree
        self.alpha_ridge = alpha_ridge
        self._basis: Optional[_BSplineBasis] = None

    def fit_basis(self, s_obs: np.ndarray) -> None:
        """Fit spline knots to observed scale values.

        Must be called before build_X / B / dB_ds.
        Called automatically by .fit() if not done explicitly.

        Parameters
        ----------
        s_obs : observed s values (positive).
        """
        s_obs = np.asarray(s_obs, float).ravel()
        if np.any(s_obs <= 0):
            raise ValueError("s_obs must be positive (basis is on log s).")
        self._basis = _BSplineBasis(n_df=self.n_df, degree=self.degree)
        self._basis.fit(np.log(s_obs))

    def _require_basis(self) -> None:
        if self._basis is None:
            raise RuntimeError("Call fit_basis(s_obs) (or fit()) before this method.")

    def build_X(self, d: np.ndarray, s: np.ndarray) -> np.ndarray:
        """X[t, k] = d_t · φₖ(log s_t),  shape (T, n_df)."""
        self._require_basis()
        d   = np.asarray(d, float).ravel()
        s   = np.asarray(s, float).ravel()
        Phi = self._basis.transform(np.log(s))   # (T, n_df)
        return Phi * d[:, None]

    def B(self, s: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """B(s) = Σₖ βₖ φₖ(log s)."""
        self._require_basis()
        s    = np.asarray(s, float).ravel()
        beta = np.asarray(beta, float).ravel()
        return self._basis.transform(np.log(s)) @ beta

    def dB_ds(self, s: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """B'(s) = (1/s) Σₖ βₖ φₖ'(log s)  [chain rule]."""
        self._require_basis()
        s    = np.asarray(s, float).ravel()
        beta = np.asarray(beta, float).ravel()
        dPhi = self._basis.derivative(np.log(s), nu=1)   # (T, n_df)
        return (dPhi @ beta) / s

    def constraint_matrices(
        self,
        s_grid: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (A_pos, A_mono) evaluated on s_grid.

        A_pos @ β ≥ ε   (positivity)
        A_mono @ β ≤ 0  (monotonicity: φₖ'(log s) @ β ≤ 0, since 1/s > 0)
        """
        self._require_basis()
        log_sg = np.log(np.asarray(s_grid, float).ravel())
        Phi    = self._basis.transform(log_sg)
        dPhi   = self._basis.derivative(log_sg, nu=1)
        return Phi, dPhi

    def fit(
        self,
        d: np.ndarray,
        s: np.ndarray,
        y: np.ndarray,
        *,
        beta_init:      Optional[np.ndarray] = None,
        use_hessian:    bool  = True,
        constrained:    bool  = False,
        s_grid:         Optional[np.ndarray] = None,
        positivity_eps: float = 1e-8,
        maxiter:        int   = 3000,
    ) -> ModelFitResult:
        """Fit B(s) = Σₖ βₖ φₖ(log s) by MLE.

        Calls fit_basis(s) internally if not already called.

        Parameters
        ----------
        constrained : enforce B(s) > 0  and  B'(s) ≤ 0  on s_grid.
        s_grid      : required when constrained=True.
        """
        d = np.asarray(d, float).ravel()
        s = np.asarray(s, float).ravel()
        y = np.asarray(y, float).ravel()
        T = len(y)

        if self._basis is None:
            self.fit_basis(s)

        X = self.build_X(d, s)

        A_pos, A_mono = None, None
        if constrained:
            if s_grid is None:
                raise ValueError("s_grid is required when constrained=True.")
            A_pos, A_mono = self.constraint_matrices(s_grid)

        beta_hat, se, diag = _fit_beta(
            X, y,
            beta_init=beta_init, alpha_ridge=self.alpha_ridge,
            use_hessian=use_hessian,
            constrained=constrained, A_pos=A_pos, A_mono=A_mono,
            positivity_eps=positivity_eps, maxiter=maxiter,
        )
        name = f"Spline(df={self.n_df}, α={self.alpha_ridge:.1g})"
        return _make_result(name, beta_hat, se, diag, T)


# ─────────────────────────────────────────────────────────────────────────────
# High-level API
# ─────────────────────────────────────────────────────────────────────────────

def fit_all_models(
    i_idx: np.ndarray,
    j_idx: np.ndarray,
    y:     np.ndarray,
    r:     np.ndarray,
    *,
    tau:            Optional[float]      = None,
    tau_grid:       Optional[np.ndarray] = None,
    tau_selection:  str   = "aic",
    spline_df:      int   = 6,
    spline_degree:  int   = 3,
    alpha_ridge:    float = 0.0,
    constrained:    bool  = False,
    s_grid:         Optional[np.ndarray] = None,
    positivity_eps: float = 1e-8,
    use_hessian:    bool  = True,
    maxiter:        int   = 3000,
) -> Dict[str, Dict]:
    """Fit LogScale, Saturating, and Spline models on the same data.

    Parameters
    ----------
    i_idx, j_idx : item index arrays.
    y            : binary outcomes in {0, 1}.
    r            : positive item ratings.
    tau          : fixed τ for SaturatingModel.  If None, tau_grid is used.
    tau_grid     : τ candidates for grid search.  If both None, a default
                   geometric grid spanning [10th, 90th pctile of s] is used.
    tau_selection: criterion for τ: 'aic', 'bic', or 'loglik'.
    spline_df    : degrees of freedom for SplineModel.
    alpha_ridge  : ridge penalty for SplineModel.
    constrained  : apply shape constraints (B > 0, B' ≤ 0) to all models.
    s_grid       : grid for shape constraints; defaults to 200 points over
                   [s.min(), s.max()] when constrained=True.
    use_hessian  : compute standard errors for all models.

    Returns
    -------
    dict with keys 'LogScale', 'Saturating', 'Spline'.
    Each value is a dict {'model': <model instance>, 'result': ModelFitResult}.
    """
    y    = np.asarray(y, float).ravel()
    d, s = compute_features(i_idx, j_idx, r)

    if constrained and s_grid is None:
        s_grid = np.linspace(float(s.min()), float(s.max()), 200)

    # ── Instantiate models ────────────────────────────────────────────────────
    logscale = LogScaleModel()

    if tau is None and tau_grid is None:
        s10, s90  = np.percentile(s, 10), np.percentile(s, 90)
        tau_grid_ = np.geomspace(max(s.min() * 0.1, 0.01), s90, 8)
    else:
        tau_grid_ = tau_grid
    saturating = SaturatingModel(tau=tau, tau_grid=tau_grid_)

    spline = SplineModel(n_df=spline_df, degree=spline_degree, alpha_ridge=alpha_ridge)

    # ── Fit ───────────────────────────────────────────────────────────────────
    kw = dict(use_hessian=use_hessian, constrained=constrained,
              s_grid=s_grid, positivity_eps=positivity_eps, maxiter=maxiter)

    out: Dict[str, Dict] = {}
    out["LogScale"]   = {"model": logscale,   "result": logscale.fit(d, s, y, **kw)}
    out["Saturating"] = {"model": saturating,
                         "result": saturating.fit(d, s, y, selection=tau_selection, **kw)}
    out["Spline"]     = {"model": spline,     "result": spline.fit(d, s, y, **kw)}
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Comparison
# ─────────────────────────────────────────────────────────────────────────────

def compare_models(fit_dict: Dict[str, Dict]) -> object:
    """Print and return a model-comparison table (sorted by AIC).

    Parameters
    ----------
    fit_dict : output of fit_all_models() or a dict of {'model':…, 'result':…}.
    """
    rows = []
    for key, bundle in fit_dict.items():
        r = bundle["result"]
        rows.append({
            "Model":     r.model_name,
            "K":         r.n_params,
            "log-lik":   round(r.log_likelihood, 4),
            "AIC":       round(r.aic, 4),
            "BIC":       round(r.bic, 4),
            "ΔAIC":      None,   # filled below
            "converged": r.diagnostics["success"],
            "|grad|":    f"{r.diagnostics['grad_norm']:.2e}",
        })

    # ΔAIC relative to the best model
    aics     = [row["AIC"] for row in rows]
    best_aic = min(aics)
    for row in rows:
        row["ΔAIC"] = round(row["AIC"] - best_aic, 4)

    try:
        import pandas as pd
        df = pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)
        print(df.to_string(index=False))
        return df
    except ImportError:
        rows_sorted = sorted(rows, key=lambda r: r["AIC"])
        header = ["Model", "K", "log-lik", "AIC", "ΔAIC", "BIC", "converged", "|grad|"]
        print("  ".join(f"{h:>12}" for h in header))
        for row in rows_sorted:
            print("  ".join(f"{str(row[h]):>12}" for h in header))
        return rows_sorted


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_B_comparison(
    fit_dict: Dict[str, Dict],
    s_obs:    np.ndarray,
    *,
    delta:    float = 0.25,
    n_grid:   int   = 300,
    ax_B=None,
    ax_W=None,
    title:    str   = "",
) -> Tuple:
    """Plot B(s) and the indecisive-region width for every fitted model.

    The indecisive-region width in d_t-space is

        width(s) = 2 c_δ / B(s),    c_δ = logit(1 − δ)

    where the "indecisive region" is { |d_t| < c_δ/B(s) } in which
    P(y=1) ∈ (δ, 1−δ).  A narrower width means the model is more decisive.

    Parameters
    ----------
    fit_dict : output of fit_all_models().
    s_obs    : observed s values (for rug marks and axis limits).
    delta    : threshold δ ∈ (0, 0.5) for the indecisive-region width.
    n_grid   : number of points on the s grid for plotting.
    ax_B, ax_W : optional existing Axes; created if None.
    title    : optional suptitle suffix.
    """
    if not (0.0 < delta < 0.5):
        raise ValueError("delta must be in (0, 0.5).")
    c_delta = float(logit(1.0 - delta))

    s_obs  = np.asarray(s_obs, float).ravel()
    s_min, s_max = float(s_obs.min()), float(s_obs.max())
    s_grid = np.linspace(s_min, s_max, n_grid)

    if ax_B is None or ax_W is None:
        fig, (ax_B, ax_W) = plt.subplots(1, 2, figsize=(13, 5))
        suptitle = f"Smooth BT models — B(s) and indecisive-region width (δ={delta})"
        if title:
            suptitle += f"  |  {title}"
        fig.suptitle(suptitle, fontsize=11)

    _COLORS = {
        "LogScale":   "steelblue",
        "Saturating": "tomato",
        "Spline":     "seagreen",
    }
    default_colors = list(_COLORS.values())

    B_all:  Dict[str, np.ndarray] = {}
    dB_all: Dict[str, np.ndarray] = {}

    for i, (key, bundle) in enumerate(fit_dict.items()):
        model  = bundle["model"]
        result = bundle["result"]
        beta   = result.beta_hat
        col    = _COLORS.get(key, default_colors[i % len(default_colors)])
        lbl    = result.model_name

        # ── Evaluate B(s) ──────────────────────────────────────────────────
        if isinstance(model, LogScaleModel):
            Bv  = model.B(s_grid, beta)
            dBv = model.dB_ds(s_grid, beta)
        elif isinstance(model, SaturatingModel):
            tau = result.diagnostics.get("tau")
            if tau is None:
                warnings.warn(f"No τ in diagnostics for '{key}'; skipping.", RuntimeWarning)
                continue
            Bv  = model.B(s_grid, beta, tau)
            dBv = model.dB_ds(s_grid, beta, tau)
        elif isinstance(model, SplineModel):
            Bv  = model.B(s_grid, beta)
            dBv = model.dB_ds(s_grid, beta)
        else:
            warnings.warn(f"Unknown model type for '{key}'; skipping.", RuntimeWarning)
            continue

        B_all[key]  = Bv
        dB_all[key] = dBv

        ax_B.plot(s_grid, Bv, lw=2, color=col, label=lbl)

        with np.errstate(divide="ignore", invalid="ignore"):
            width = np.where(Bv > 1e-6, 2.0 * c_delta / Bv, np.nan)
        ax_W.plot(s_grid, width, lw=2, color=col, label=lbl)

    # ── Rug of observed s values (added after all lines so y-limits are set) ──
    for ax, ylabel, panel_title in [
        (ax_B, "B(s)", "Discriminability B(s)"),
        (ax_W, f"width(s) = 2·c_δ / B(s)  [δ={delta}]", "Indecisive-region width"),
    ]:
        y_lo = ax.get_ylim()[0]
        ax.plot(s_obs, np.full_like(s_obs, y_lo), "|",
                color="gray", alpha=0.2, ms=5, label="obs. s")
        ax.axhline(0.0, color="black", lw=0.8, ls="--", alpha=0.4)
        ax.set_xlabel("s = r_i + r_j", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(panel_title, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    return ax_B, ax_W


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic-data demo
# ─────────────────────────────────────────────────────────────────────────────

def synthetic_demo(
    seed:      int   = 0,
    N:         int   = 50,
    T:         int   = 5000,
    tau_true:  float = 2.0,
    beta_true: Optional[np.ndarray] = None,
    spline_df: int   = 6,
    constrained: bool = False,
) -> Dict[str, Dict]:
    """Synthetic-data demo with an early-declining true B(s).

    Data-generating process (saturating truth):
        r_k ~ Uniform(0.5, 5.0)
        B(s) = β₀ + β₁(1 − e^{−s/τ_true})
        y_t ~ Bernoulli(σ(B(s_t) d_t))

    Fits all three models and prints a comparison table + plot.

    Parameters
    ----------
    beta_true : true [β₀, β₁].  Defaults to [2.0, −1.5], which gives
                B(0) = 2, B(∞) = 0.5 — an early drop.
    """
    rng = np.random.default_rng(seed)

    if beta_true is None:
        beta_true = np.array([2.0, -1.5])
    beta_true = np.asarray(beta_true, float).ravel()
    assert beta_true.shape == (2,)

    # ── Generate data ─────────────────────────────────────────────────────────
    r     = rng.uniform(0.5, 5.0, size=N)
    i_idx = rng.integers(0, N, size=T)
    j_idx = rng.integers(0, N, size=T)
    same  = i_idx == j_idx
    j_idx[same] = (j_idx[same] + 1) % N

    d, s = compute_features(i_idx, j_idx, r)

    sat_true = SaturatingModel(tau=tau_true)
    X_true   = sat_true.build_X(d, s, tau_true)
    p_true   = expit(X_true @ beta_true)
    y        = rng.binomial(1, p_true).astype(float)

    print("══ Synthetic demo ══════════════════════════════════════════")
    print(f"N={N} items, T={T} trials")
    print(f"True model: Saturating  τ={tau_true}  β={beta_true}")
    print(f"B(s_min={s.min():.2f}) = {sat_true.B(s.min(), beta_true, tau_true):.3f}  "
          f"B(s_max={s.max():.2f}) = {sat_true.B(s.max(), beta_true, tau_true):.3f}")
    print(f"Fraction y=1: {y.mean():.3f}")
    print()

    # ── Fit ───────────────────────────────────────────────────────────────────
    s10, s90 = np.percentile(s, 10), np.percentile(s, 90)
    tau_grid = np.geomspace(max(s.min() * 0.05, 0.01), s90 * 2, 12)

    out = fit_all_models(
        i_idx, j_idx, y, r,
        tau_grid=tau_grid,
        spline_df=spline_df,
        alpha_ridge=1e-3,
        constrained=constrained,
        use_hessian=True,
    )

    print()
    for key, bundle in out.items():
        print(bundle["result"].summary())
        print()

    print("══ Model comparison ════════════════════════════════════════")
    compare_models(out)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, (ax_B, ax_W) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Synthetic demo: true B(s) vs fitted models", fontsize=11)

    s_grid = np.linspace(float(s.min()), float(s.max()), 300)
    B_true_vals = sat_true.B(s_grid, beta_true, tau_true)
    ax_B.plot(s_grid, B_true_vals, "k--", lw=2,
              label=f"True (Saturating, τ={tau_true})")
    c_delta = float(logit(0.75))
    with np.errstate(divide="ignore", invalid="ignore"):
        ax_W.plot(s_grid,
                  np.where(B_true_vals > 1e-6, 2.0 * c_delta / B_true_vals, np.nan),
                  "k--", lw=2, label=f"True (Saturating, τ={tau_true})")

    plot_B_comparison(out, s_obs=s, delta=0.25, ax_B=ax_B, ax_W=ax_W)

    plt.tight_layout()
    plt.savefig("smooth_bt_models_demo.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved smooth_bt_models_demo.png")

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    synthetic_demo()
