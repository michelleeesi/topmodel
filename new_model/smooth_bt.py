"""
smooth_bt.py
────────────
MLE for the two-parameter smooth Bradley–Terry model

    P(y_t = 1) = σ( (β₀ + β₁ s_t) · d_t )

where
    d_t = log r_{i_t} − log r_{j_t}   log-ratio of positive BT ratings
    s_t = r_{i_t} + r_{j_t}           total quality of the pair

This is equivalent to logistic regression on two features

    x_{t,0} = d_t
    x_{t,1} = s_t · d_t

so the linear predictor is  η_t = β₀ d_t + β₁ s_t d_t = B(s_t) d_t
with  B(s) = β₀ + β₁ s.

Standard BT (constant discriminability) is the special case β₁ = 0.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import LinAlgError
from scipy.optimize import LinearConstraint, minimize
from scipy.special import expit


# ─────────────────────────────────────────────────────────────────────────────
# Design matrix
# ─────────────────────────────────────────────────────────────────────────────

def build_design_matrix(
    r_i: np.ndarray,
    r_j: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the (T, 2) design matrix for the linear-B pairwise model.

    Columns:
        X[:, 0] = d_t  =  log(r_i) − log(r_j)
        X[:, 1] = s_t · d_t,   where  s_t = r_i + r_j

    Parameters
    ----------
    r_i : array-like, shape (T,)
        Strictly positive BT ratings for the focal (left / i) item.
    r_j : array-like, shape (T,)
        Strictly positive BT ratings for the comparison (right / j) item.

    Returns
    -------
    X : ndarray, shape (T, 2)
        Design matrix.
    d : ndarray, shape (T,)
        Log-ratio features  log(r_i) − log(r_j).
    s : ndarray, shape (T,)
        Sum features  r_i + r_j.

    Raises
    ------
    ValueError
        If any rating is non-positive or non-finite, or shapes mismatch.
    """
    r_i = np.asarray(r_i, dtype=float).ravel()
    r_j = np.asarray(r_j, dtype=float).ravel()

    if r_i.shape != r_j.shape:
        raise ValueError(
            f"Shape mismatch: r_i has {r_i.shape}, r_j has {r_j.shape}."
        )
    if not (np.all(np.isfinite(r_i)) and np.all(np.isfinite(r_j))):
        raise ValueError("All ratings must be finite.")
    if np.any(r_i <= 0.0) or np.any(r_j <= 0.0):
        raise ValueError(
            "All ratings must be strictly positive (required for the log ratio)."
        )

    d = np.log(r_i) - np.log(r_j)
    s = r_i + r_j
    X = np.column_stack([d, s * d])
    return X, d, s


# ─────────────────────────────────────────────────────────────────────────────
# Negative log-likelihood, gradient, Hessian
# ─────────────────────────────────────────────────────────────────────────────

def neg_log_likelihood(beta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """Numerically stable negative log-likelihood for logistic regression.

    NLL(β) = Σ_t [ logaddexp(0, z_t) − y_t · z_t ]
           = Σ_t [ log(1 + exp(z_t)) − y_t · z_t ]

    where  z_t = X[t] @ β.  The logaddexp form is exact for all z.

    Parameters
    ----------
    beta : ndarray, shape (2,)
    X    : ndarray, shape (T, 2)
    y    : ndarray, shape (T,)  binary outcomes in {0, 1}

    Returns
    -------
    float
    """
    z = X @ beta
    return float(np.sum(np.logaddexp(0.0, z) - y * z))


def nll_gradient(beta: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Gradient of the negative log-likelihood.

    ∇NLL(β) = X^T (σ(Xβ) − y)

    Parameters
    ----------
    beta : ndarray, shape (2,)
    X    : ndarray, shape (T, 2)
    y    : ndarray, shape (T,)

    Returns
    -------
    ndarray, shape (2,)
    """
    p = expit(X @ beta)
    return X.T @ (p - y)


def nll_hessian(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Observed Hessian of the negative log-likelihood (positive semi-definite).

    H(β) = X^T diag(p ⊙ (1 − p)) X,   p = σ(Xβ)

    Each weight  p_t(1 − p_t) ∈ (0, ¼].

    Parameters
    ----------
    beta : ndarray, shape (2,)
    X    : ndarray, shape (T, 2)

    Returns
    -------
    ndarray, shape (2, 2)
    """
    p = expit(X @ beta)
    w = p * (1.0 - p)
    return (X * w[:, None]).T @ X


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FitResult:
    """Return type for :func:`fit`.

    Attributes
    ----------
    beta_hat : ndarray, shape (2,)
        MLE estimates [β₀, β₁].
    standard_errors : ndarray, shape (2,) or None
        Square-root diagonal of the observed Fisher information inverse.
        None when the Hessian is singular or ill-conditioned.
    fitted_probabilities : ndarray, shape (T,)
        σ(B(s_t) · d_t) evaluated at ``beta_hat``.
    diagnostics : dict
        Optimizer and numerical diagnostics.  Keys:

        ``success`` bool
            Whether the optimizer reported convergence.
        ``message`` str
            Optimizer status message.
        ``n_iter`` int or None
            Number of optimizer iterations.
        ``nfev`` / ``njev`` int or None
            Function / Jacobian evaluations.
        ``fun`` float
            Objective value (NLL) at the solution.
        ``log_lik`` float
            Log-likelihood at the solution (= −fun).
        ``grad_norm`` float
            Euclidean norm of the gradient at the solution.
        ``X_rank`` int
            Numerical rank of the design matrix.
        ``X_cond`` float
            Condition number of the design matrix.
        ``H_cond`` float or None
            Condition number of the Hessian at the solution.
        ``constrained`` bool
        ``method`` str
            scipy optimizer method used.
        ``optimizer_result`` OptimizeResult
            Full scipy result object.
        ``X`` ndarray, shape (T, 2)
        ``d`` ndarray, shape (T,)
        ``s`` ndarray, shape (T,)
    """

    beta_hat: np.ndarray
    standard_errors: Optional[np.ndarray]
    fitted_probabilities: np.ndarray
    diagnostics: Dict


# ─────────────────────────────────────────────────────────────────────────────
# Main fit function
# ─────────────────────────────────────────────────────────────────────────────

def fit(
    y: np.ndarray,
    r_i: np.ndarray,
    r_j: np.ndarray,
    *,
    beta_init: Optional[np.ndarray] = None,
    use_hessian: bool = True,
    constrained: bool = False,
    s_grid: Optional[np.ndarray] = None,
    positivity_eps: float = 1e-8,
    warn_rank_cond: float = 1e12,
    warn_prob_eps: float = 1e-8,
    maxiter: int = 2000,
) -> FitResult:
    """Fit  P(y=1) = σ((β₀ + β₁ s_t) d_t)  by maximum likelihood.

    Parameters
    ----------
    y : array-like, shape (T,)
        Binary outcomes in {0, 1}.  y_t = 1 means item i_t was preferred.
    r_i : array-like, shape (T,)
        Strictly positive BT ratings for the focal item.
    r_j : array-like, shape (T,)
        Strictly positive BT ratings for the comparison item.
    beta_init : array-like, shape (2,), optional
        Initial point [β₀, β₁].  Defaults to [0, 0].
    use_hessian : bool
        Compute standard errors from the observed Fisher information.
    constrained : bool
        Enforce the shape constraints

        * β₁ ≤ 0              (B is non-increasing in s)
        * β₀ + β₁ s_m ≥ positivity_eps  for all  s_m ∈ s_grid

        Requires ``s_grid`` to be supplied.
    s_grid : array-like, optional
        Grid of s values for positivity constraints.  Required when
        ``constrained=True``.  A sensible default is
        ``np.linspace(s.min(), s.max(), 200)`` where s comes from the data.
    positivity_eps : float
        Minimum allowed value of B(s) on the constraint grid (default 1e-8).
    warn_rank_cond : float
        Condition-number threshold; a RuntimeWarning is issued if exceeded.
    warn_prob_eps : float
        Warn if any fitted probability is within this of 0 or 1 (near-separation).
    maxiter : int
        Maximum optimizer iterations.

    Returns
    -------
    FitResult
        See :class:`FitResult`.

    Notes
    -----
    The unconstrained problem uses BFGS with exact gradient.
    The constrained problem uses trust-constr with exact gradient and Hessian,
    which gives reliable convergence even near active constraints.

    The observed Fisher information is

        I(β̂) = X^T diag(p̂ ⊙ (1 − p̂)) X

    and  Cov(β̂) ≈ I(β̂)⁻¹  gives the standard errors.

    Raises
    ------
    ValueError
        For invalid inputs (non-binary y, non-positive ratings, missing s_grid).
    """
    # ── Input coercion & validation ───────────────────────────────────────────
    y = np.asarray(y, dtype=float).ravel()
    if np.any((y != 0.0) & (y != 1.0)):
        raise ValueError("y must be binary in {0, 1}.")

    X, d, s = build_design_matrix(r_i, r_j)   # validates ratings internally
    T = len(y)
    if X.shape[0] != T:
        raise ValueError(
            f"Length mismatch: y has {T} entries, but r_i/r_j have {X.shape[0]}."
        )

    # ── Design-matrix numerical health ────────────────────────────────────────
    rank = int(np.linalg.matrix_rank(X))
    if rank < 2:
        warnings.warn(
            f"Design matrix is rank-deficient (rank={rank}). "
            "Estimates may not be unique.",
            RuntimeWarning,
        )

    sv = np.linalg.svd(X, full_matrices=False, compute_uv=False)
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")
    if not np.isfinite(cond) or cond > warn_rank_cond:
        warnings.warn(
            f"Design matrix is ill-conditioned (cond ≈ {cond:.2e}). "
            "Estimates and standard errors may be unreliable.",
            RuntimeWarning,
        )

    # ── Separation check ──────────────────────────────────────────────────────
    for k in range(2):
        col = X[:, k]
        if (y == 1).any() and (y == 0).any():
            if np.all(col[y == 1] > 0) and np.all(col[y == 0] < 0):
                warnings.warn(
                    f"Column {k} of the design matrix may perfectly separate classes. "
                    "MLE may be at ±∞; consider L2 regularisation.",
                    RuntimeWarning,
                )

    # ── Initial point ─────────────────────────────────────────────────────────
    if beta_init is None:
        beta_init = np.zeros(2, dtype=float)
    else:
        beta_init = np.asarray(beta_init, dtype=float).ravel()
        if beta_init.shape != (2,):
            raise ValueError("beta_init must have shape (2,).")

    # ── Objective / gradient / Hessian closures ───────────────────────────────
    def fun(b: np.ndarray) -> float:
        return neg_log_likelihood(b, X, y)

    def jac(b: np.ndarray) -> np.ndarray:
        return nll_gradient(b, X, y)

    def hess(b: np.ndarray) -> np.ndarray:
        return nll_hessian(b, X)

    # ── Solve ─────────────────────────────────────────────────────────────────
    if not constrained:
        res = minimize(
            fun, beta_init,
            method="L-BFGS-B",
            jac=jac,
            options={"maxiter": maxiter, "ftol": 1e-15, "gtol": 1e-8},
        )
        method_used = "L-BFGS-B"

    else:
        if s_grid is None:
            raise ValueError("s_grid must be provided when constrained=True.")
        s_grid = np.asarray(s_grid, dtype=float).ravel()
        if s_grid.size == 0 or not np.all(np.isfinite(s_grid)):
            raise ValueError("s_grid must be a non-empty finite array.")

        # Constraint 1: β₀ + β₁ sₘ ≥ positivity_eps  for each grid point
        #   A_pos @ β ≥ lb_pos   with  A_pos[m] = [1, s_m]
        A_pos = np.column_stack([np.ones_like(s_grid), s_grid])  # (G, 2)
        lb_pos = np.full(len(s_grid), float(positivity_eps))
        ub_pos = np.full(len(s_grid), np.inf)

        # Constraint 2: β₁ ≤ 0
        A_mono = np.array([[0.0, 1.0]])
        lb_mono = np.array([-np.inf])
        ub_mono = np.array([0.0])

        A = np.vstack([A_pos, A_mono])
        lb = np.concatenate([lb_pos, lb_mono])
        ub = np.concatenate([ub_pos, ub_mono])

        res = minimize(
            fun, beta_init,
            method="trust-constr",
            jac=jac,
            hess=hess,
            constraints=[LinearConstraint(A, lb, ub)],
            options={"maxiter": maxiter, "verbose": 0},
        )
        method_used = "trust-constr"

    if not res.success:
        warnings.warn(
            f"Optimiser did not converge: {res.message}",
            RuntimeWarning,
        )

    beta_hat = np.asarray(res.x, dtype=float)

    # ── Post-fit separation heuristic ─────────────────────────────────────────
    if np.any(np.abs(beta_hat) > 50.0):
        warnings.warn(
            "Some |β̂| > 50.  This often signals near-complete separation; "
            "fitted probabilities near 0/1 should be treated with caution.",
            RuntimeWarning,
        )

    z_hat = X @ beta_hat
    p_hat = expit(z_hat)

    n_extreme = int(np.sum((p_hat < warn_prob_eps) | (p_hat > 1.0 - warn_prob_eps)))
    if n_extreme > 0:
        warnings.warn(
            f"Fitted probabilities are extremely close to 0 or 1 for "
            f"{n_extreme}/{T} trials ({n_extreme/T:.1%}). "
            "This may indicate near-separation or extreme logits.",
            RuntimeWarning,
        )

    # ── Standard errors from observed Fisher information ──────────────────────
    se: Optional[np.ndarray] = None
    H_cond: Optional[float] = None

    if use_hessian:
        try:
            H = nll_hessian(beta_hat, X)
            sv_H = np.linalg.svd(H, compute_uv=False)
            H_cond = float(sv_H[0] / sv_H[-1]) if sv_H[-1] > 0 else float("inf")

            if np.isfinite(H_cond) and H_cond < 1e16:
                cov = np.linalg.inv(H)
            else:
                warnings.warn(
                    f"Hessian is ill-conditioned (cond ≈ {H_cond:.2e}); "
                    "using pseudo-inverse for standard errors.",
                    RuntimeWarning,
                )
                cov = np.linalg.pinv(H)

            var = np.diag(cov)
            if np.any(var < 0):
                warnings.warn(
                    "Negative variance(s) on the Hessian diagonal; "
                    "those standard errors are set to NaN.",
                    RuntimeWarning,
                )
                var = np.where(var >= 0, var, np.nan)
            se = np.sqrt(var)

        except LinAlgError:
            warnings.warn(
                "Hessian inversion failed; standard errors unavailable.",
                RuntimeWarning,
            )

    # ── Diagnostics bundle ────────────────────────────────────────────────────
    nll_val = neg_log_likelihood(beta_hat, X, y)
    grad_norm = float(np.linalg.norm(nll_gradient(beta_hat, X, y)))

    diagnostics: Dict = {
        "success": bool(res.success),
        "status": int(res.status) if hasattr(res, "status") and res.status is not None else None,
        "message": str(res.message),
        "n_iter": int(res.nit) if hasattr(res, "nit") and res.nit is not None else None,
        "nfev": int(res.nfev) if hasattr(res, "nfev") else None,
        "njev": int(res.njev) if hasattr(res, "njev") else None,
        "fun": float(res.fun),
        "log_lik": float(-nll_val),
        "grad_norm": grad_norm,
        "X_rank": rank,
        "X_cond": float(cond),
        "H_cond": H_cond,
        "constrained": bool(constrained),
        "method": method_used,
        "optimizer_result": res,
        "X": X,
        "d": d,
        "s": s,
    }

    return FitResult(
        beta_hat=beta_hat,
        standard_errors=se,
        fitted_probabilities=p_hat,
        diagnostics=diagnostics,
    )


# ─────────────────────────────────────────────────────────────────────────────
# B(s) helpers
# ─────────────────────────────────────────────────────────────────────────────

def B(s: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Evaluate the discriminability function  B(s) = β₀ + β₁ s.

    Parameters
    ----------
    s : array-like
        Scale values  s_t = r_i + r_j.
    beta : array-like, shape (2,)
        Coefficients [β₀, β₁].

    Returns
    -------
    ndarray  same shape as s
    """
    s = np.asarray(s, dtype=float)
    beta = np.asarray(beta, dtype=float).ravel()
    if beta.shape != (2,):
        raise ValueError("beta must have shape (2,).")
    return beta[0] + beta[1] * s


def plot_B(
    s_range: np.ndarray,
    beta: np.ndarray,
    *,
    ax=None,
    label: Optional[str] = None,
    show_derivative: bool = False,
    title: str = "Discriminability function B(s)",
    **line_kw,
):
    """Plot  B(s) = β₀ + β₁ s  over a grid of s values.

    Parameters
    ----------
    s_range : array-like
        Values of s at which to evaluate B.
    beta : array-like, shape (2,)
        Coefficients [β₀, β₁].
    ax : matplotlib Axes, optional
        Existing Axes to draw on; created if None.
    label : str, optional
        Legend label.  Defaults to the equation string.
    show_derivative : bool
        If True, annotate with  B'(s) = β₁  on a twin y-axis.
    title : str
        Axes title.
    **line_kw
        Extra keyword arguments forwarded to ``ax.plot``.

    Returns
    -------
    ax : primary matplotlib Axes
    """
    s_arr = np.asarray(s_range, dtype=float)
    beta = np.asarray(beta, dtype=float).ravel()
    if beta.shape != (2,):
        raise ValueError("beta must have shape (2,).")

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    _label = label or f"B(s) = {beta[0]:.4g} + {beta[1]:.4g}·s"
    ax.plot(s_arr, B(s_arr, beta), lw=2, label=_label, **line_kw)
    ax.axhline(0.0, color="black", lw=0.8, ls="--", alpha=0.4)
    ax.set_xlabel("s = r_i + r_j", fontsize=11)
    ax.set_ylabel("B(s)", fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)

    if show_derivative:
        ax2 = ax.twinx()
        ax2.axhline(
            beta[1], color="tomato", lw=1.5, ls=":",
            label=f"B'(s) = {beta[1]:.4g}",
        )
        ax2.set_ylabel("B'(s) = β₁", color="tomato", fontsize=10)
        ax2.tick_params(axis="y", labelcolor="tomato")
        ax2.legend(fontsize=8, loc="upper right")

    return ax


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic-data example
# ─────────────────────────────────────────────────────────────────────────────

def synthetic_demo(
    seed: int = 42,
    T: int = 5000,
    beta_true: Optional[np.ndarray] = None,
    constrained: bool = True,
) -> Tuple[FitResult, Optional[FitResult]]:
    """Demonstrate the model on synthetic data with a known B(s).

    Data-generating process
    -----------------------
    * r_i, r_j ~ LogNormal(0, 1)   →  ratings positive with wide dynamic range
    * Default β_true = [1.0, −2e-4]  →  B(s) = 1 − 0.0002 s  (positive, decreasing)
    * y_t ~ Bernoulli(σ(B(s_t) d_t))

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    T : int
        Number of comparison trials.
    beta_true : array-like, shape (2,), optional
        True [β₀, β₁].  Defaults to [1.0, −2e-4].
    constrained : bool
        If True, also run the constrained fit (β₁ ≤ 0, B(s) > 0) and overlay
        it on the plot.

    Returns
    -------
    out_unc : FitResult
        Unconstrained MLE.
    out_con : FitResult or None
        Constrained MLE if ``constrained=True``, else None.
    """
    rng = np.random.default_rng(seed)

    if beta_true is None:
        beta_true = np.array([1.0, -2e-4])
    beta_true = np.asarray(beta_true, dtype=float).ravel()
    if beta_true.shape != (2,):
        raise ValueError("beta_true must have shape (2,).")

    # ── Simulate data ─────────────────────────────────────────────────────────
    r_i = rng.lognormal(mean=0.0, sigma=1.0, size=T)
    r_j = rng.lognormal(mean=0.0, sigma=1.0, size=T)
    X_true, _, s_true = build_design_matrix(r_i, r_j)

    p_true = expit(X_true @ beta_true)
    y = rng.binomial(1, p_true).astype(float)

    print(f"Synthetic demo: T = {T} comparisons")
    print(f"True β:       β₀ = {beta_true[0]:.4g},  β₁ = {beta_true[1]:.4g}")
    print(f"Fraction y=1: {y.mean():.3f}")

    # ── Unconstrained fit ─────────────────────────────────────────────────────
    out_unc = fit(y, r_i, r_j, use_hessian=True)
    bh, se = out_unc.beta_hat, out_unc.standard_errors
    se_str = (
        f"(SE = {se[0]:.4g})" if se is not None else "(SE unavailable)"
    )
    print(f"\nUnconstrained:  β̂₀ = {bh[0]:.4g} {se_str},  "
          f"β̂₁ = {bh[1]:.4g} "
          f"{'(SE = ' + f'{se[1]:.4g})' if se is not None else ''}")
    print(f"  log-lik = {out_unc.diagnostics['log_lik']:.4f}  "
          f"|grad| = {out_unc.diagnostics['grad_norm']:.2e}  "
          f"converged = {out_unc.diagnostics['success']}")

    # ── Constrained fit ───────────────────────────────────────────────────────
    out_con = None
    if constrained:
        s_grid = np.linspace(float(s_true.min()), float(s_true.max()), 200)
        out_con = fit(y, r_i, r_j, constrained=True, s_grid=s_grid)
        bh_c = out_con.beta_hat
        print(f"\nConstrained:    β̂₀ = {bh_c[0]:.4g},  β̂₁ = {bh_c[1]:.4g}  "
              f"(β₁ ≤ 0, B(s) > 0 on grid)")
        print(f"  log-lik = {out_con.diagnostics['log_lik']:.4f}  "
              f"converged = {out_con.diagnostics['success']}")

    # ── Plot B(s) ─────────────────────────────────────────────────────────────
    s_range = np.linspace(float(s_true.min()), float(s_true.max()), 300)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(s_range, B(s_range, beta_true), "k--", lw=1.5, label="True B(s)")
    plot_B(s_range, out_unc.beta_hat, ax=ax,
           label="Fitted (unconstrained)", color="steelblue")
    if out_con is not None:
        plot_B(s_range, out_con.beta_hat, ax=ax,
               label="Fitted (constrained)", color="tomato")
    ax.set_title("Synthetic demo: true vs fitted B(s)")
    ax.legend()
    plt.tight_layout()
    plt.show()

    return out_unc, out_con


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    synthetic_demo(constrained=True)
