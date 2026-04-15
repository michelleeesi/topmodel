"""
smooth_bt.py
────────────
MLE for a smooth Bradley–Terry model where discriminability depends on the
"scale" s = r_i + r_j of a comparison.

Model
─────
Each item k has a positive scalar rating r[k] > 0.
For a comparison between item i (focal) and item j (comparison):

    d_t = log(r[i_t]) - log(r[j_t])        log-ratio of ratings
    s_t = r[i_t] + r[j_t]                  scale of the comparison

A smooth discriminability function parameterised by beta:

    B_beta(s) = sum_{k=0}^{K-1}  beta_k * phi_k(s)

gives the logistic model:

    P(y_t = 1 | i_t, j_t) = sigmoid( B_beta(s_t) * d_t )

where y_t = 1 means focal item i_t is preferred over j_t.

Design matrix (reduces to standard logistic regression):

    X[t, k] = d_t * phi_k(s_t)

so  eta_t = X[t] @ beta = B_beta(s_t) * d_t.

Log-likelihood:
    ell(beta) = sum_t [ y_t * eta_t - log(1 + exp(eta_t)) ]

Standard BT (constant discriminability) is recovered with K=1, phi_0(s)=1.
"""

import warnings
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy.optimize import minimize, LinearConstraint
from scipy.special import expit  # numerically stable sigmoid


# ─────────────────────────────────────────────────────────────────────────────
# Basis function factories
# ─────────────────────────────────────────────────────────────────────────────

def poly_basis(degree):
    """
    Return (phi, phi_prime) for the monomial phi(s) = s^degree.

    Examples
    --------
    >>> phi, dphi = poly_basis(2)
    >>> phi(3.0), dphi(3.0)   # 9.0, 6.0
    """
    def phi(s):
        return np.asarray(s, dtype=float) ** degree

    def phi_prime(s):
        s = np.asarray(s, dtype=float)
        return degree * s ** (degree - 1) if degree > 0 else np.zeros_like(s)

    return phi, phi_prime


def make_poly_basis(max_degree):
    """
    Polynomial basis: phi_k(s) = s^k for k = 0, ..., max_degree.

    Returns
    -------
    phi_list  : list of callables
    dphi_list : list of derivative callables
    """
    pairs = [poly_basis(k) for k in range(max_degree + 1)]
    phi_list, dphi_list = zip(*pairs)
    return list(phi_list), list(dphi_list)


# ─────────────────────────────────────────────────────────────────────────────
# Core model helpers
# ─────────────────────────────────────────────────────────────────────────────

def B(s, beta, phi_list):
    """
    Evaluate the discriminability function B_beta(s).

    B_beta(s) = sum_k  beta[k] * phi_k(s)

    Parameters
    ----------
    s        : scalar or array-like
    beta     : array, shape (K,)
    phi_list : list of K callables

    Returns
    -------
    float or ndarray
    """
    s = np.asarray(s, dtype=float)
    return sum(beta[k] * phi(s) for k, phi in enumerate(phi_list))


def build_design_matrix(r_i_idx, r_j_idx, r, phi_list):
    """
    Build the logistic-regression design matrix X.

    X[t, k] = d_t * phi_k(s_t)

    where
        d_t = log(r[i_t]) - log(r[j_t])    log-ratio
        s_t = r[i_t] + r[j_t]              scale

    Parameters
    ----------
    r_i_idx  : int array, shape (T,)   focal item indices
    r_j_idx  : int array, shape (T,)   comparison item indices
    r        : float array, shape (N,)  item ratings, all > 0
    phi_list : list of K callables

    Returns
    -------
    X : float array, shape (T, K)
    d : float array, shape (T,)    log-ratio features
    s : float array, shape (T,)    scale features
    """
    r = np.asarray(r, dtype=float)
    if np.any(r <= 0):
        raise ValueError("All item ratings r must be strictly positive.")

    r_i = r[r_i_idx]
    r_j = r[r_j_idx]

    d = np.log(r_i) - np.log(r_j)
    s = r_i + r_j

    T, K = len(d), len(phi_list)
    X = np.empty((T, K), dtype=float)
    for k, phi in enumerate(phi_list):
        X[:, k] = d * phi(s)

    return X, d, s


# ─────────────────────────────────────────────────────────────────────────────
# Negative log-likelihood, gradient, Hessian
# ─────────────────────────────────────────────────────────────────────────────

def neg_log_likelihood(beta, X, y):
    """
    Negative log-likelihood.

    NLL = -sum_t [ y_t * eta_t - logaddexp(0, eta_t) ]
        = -sum_t [ y_t * eta_t - log(1 + exp(eta_t)) ]

    logaddexp(0, eta) is numerically stable for all eta.
    """
    eta = X @ beta
    return -np.sum(y * eta - np.logaddexp(0.0, eta))


def nll_gradient(beta, X, y):
    """
    Gradient of NLL w.r.t. beta.

    grad = X^T (p - y)     where p_t = sigmoid(eta_t)
    """
    p = expit(X @ beta)
    return X.T @ (p - y)


def nll_hessian(beta, X, y):
    """
    Hessian of NLL w.r.t. beta.

    H = X^T diag(p * (1 - p)) X     (positive semi-definite)
    """
    p = expit(X @ beta)
    w = p * (1.0 - p)
    return (X * w[:, None]).T @ X


# ─────────────────────────────────────────────────────────────────────────────
# Constraint matrix builders
# ─────────────────────────────────────────────────────────────────────────────

def _eval_basis_on_grid(s_grid, func_list):
    """
    Evaluate a list of functions on a grid.

    Returns M[g, k] = func_list[k](s_grid[g]), shape (G, K).
    """
    G, K = len(s_grid), len(func_list)
    M = np.empty((G, K), dtype=float)
    for k, f in enumerate(func_list):
        M[:, k] = f(s_grid)
    return M


# ─────────────────────────────────────────────────────────────────────────────
# Main fit function
# ─────────────────────────────────────────────────────────────────────────────

def fit(
    r_i_idx,
    r_j_idx,
    y,
    r,
    phi_list,
    dphi_list=None,
    beta0=None,
    constrained=False,
    s_grid=None,
    B_min=1e-6,
    method=None,
    options=None,
):
    """
    Fit the smooth Bradley–Terry model by maximum likelihood.

    Parameters
    ----------
    r_i_idx     : int array (T,)    focal item indices
    r_j_idx     : int array (T,)    comparison item indices
    y           : binary array (T,) 1 = focal item wins
    r           : float array (N,)  item ratings (must be > 0)
    phi_list    : list of K callables phi_k(s)
    dphi_list   : list of K callables phi_k'(s)
                  Required only when constrained=True and you want B' <= 0.
    beta0       : initial beta, shape (K,). Defaults to zeros.
    constrained : bool
                  Enforce B_beta(s) >= B_min and (if dphi_list given)
                  B_beta'(s) <= 0 on s_grid.
    s_grid      : 1-D array of s values used for shape constraints.
                  Defaults to 50 evenly-spaced points over observed s range.
    B_min       : lower bound for B_beta on s_grid (default 1e-6).
    method      : scipy optimizer. Default: 'L-BFGS-B' (unconstrained),
                  'SLSQP' (constrained).
    options     : dict forwarded to scipy.optimize.minimize.

    Returns
    -------
    dict with keys
        beta_hat  ndarray (K,)    fitted coefficients
        se        ndarray (K,) or None   standard errors
        probs     ndarray (T,)    fitted P(y=1)
        log_lik   float           final log-likelihood
        nll       float           final negative log-likelihood
        status    bool            optimizer success
        message   str             optimizer message
        result    OptimizeResult  full scipy result
        X         ndarray (T, K)  design matrix
        d         ndarray (T,)    log-ratio features
        s         ndarray (T,)    scale features
    """
    # ── Input validation ─────────────────────────────────────────────────────
    r_i_idx = np.asarray(r_i_idx, dtype=int)
    r_j_idx = np.asarray(r_j_idx, dtype=int)
    y       = np.asarray(y,       dtype=float)
    r       = np.asarray(r,       dtype=float)

    if np.any(r <= 0):
        raise ValueError("All item ratings r must be > 0.")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("y must be binary (0 or 1).")

    K = len(phi_list)
    X, d, s = build_design_matrix(r_i_idx, r_j_idx, r, phi_list)

    # ── Rank check ───────────────────────────────────────────────────────────
    rank = np.linalg.matrix_rank(X)
    if rank < K:
        warnings.warn(
            f"Design matrix is rank-deficient (rank {rank} < {K} columns). "
            "Coefficient estimates may not be unique.",
            RuntimeWarning,
        )

    # ── Separation check: any single column perfectly separates classes? ─────
    for k in range(K):
        xk = X[:, k]
        pos_ok = np.all(xk[y == 1] > 0) if (y == 1).any() else False
        neg_ok = np.all(xk[y == 0] < 0) if (y == 0).any() else False
        if pos_ok and neg_ok:
            warnings.warn(
                f"Column {k} of design matrix may cause complete separation. "
                "MLE may be at infinity; consider L2 regularisation.",
                RuntimeWarning,
            )

    beta0 = np.zeros(K) if beta0 is None else np.asarray(beta0, dtype=float)

    obj  = lambda b: neg_log_likelihood(b, X, y)
    grad = lambda b: nll_gradient(b, X, y)

    # ── Unconstrained fit ────────────────────────────────────────────────────
    if not constrained:
        _method  = method  or "L-BFGS-B"
        _options = options or {"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8}
        result = minimize(obj, beta0, jac=grad, method=_method, options=_options)

    # ── Constrained fit ──────────────────────────────────────────────────────
    else:
        if s_grid is None:
            s_grid = np.linspace(s.min(), s.max(), 50)
        s_grid = np.asarray(s_grid, dtype=float)

        # Phi[g, k] = phi_k(s_grid[g])
        Phi = _eval_basis_on_grid(s_grid, phi_list)

        constraints = [
            # B_beta(s) >= B_min  on s_grid: Phi @ beta >= B_min
            LinearConstraint(Phi, lb=B_min, ub=np.inf),
        ]

        if dphi_list is not None:
            # B_beta'(s) <= 0  on s_grid: Phi_prime @ beta <= 0
            Phi_prime = _eval_basis_on_grid(s_grid, dphi_list)
            constraints.append(LinearConstraint(Phi_prime, lb=-np.inf, ub=0.0))
        else:
            warnings.warn(
                "dphi_list not provided; B'(s) <= 0 constraint is skipped.",
                RuntimeWarning,
            )

        _method  = method  or "SLSQP"
        _options = options or {"maxiter": 2000, "ftol": 1e-12}
        result = minimize(
            obj, beta0, jac=grad,
            method=_method, constraints=constraints, options=_options,
        )

    beta_hat = result.x

    # ── Standard errors from observed Fisher information matrix ───────────────
    # I(beta) = X^T diag(p*(1-p)) X;  Var(beta_hat) ≈ I(beta_hat)^{-1}
    se = None
    try:
        H = nll_hessian(beta_hat, X, y)
        # Cholesky: succeeds iff H is positive definite
        L = la.cholesky(H, lower=True)
        # inv(H) = (L^{-1})^T L^{-1}
        Linv = la.solve_triangular(L, np.eye(K), lower=True)
        cov  = Linv.T @ Linv
        se   = np.sqrt(np.diag(cov))
    except la.LinAlgError:
        warnings.warn(
            "Hessian is not positive definite at beta_hat; "
            "standard errors are unavailable. "
            "The model may be unidentified or near separation.",
            RuntimeWarning,
        )

    # ── Post-fit separation warning ───────────────────────────────────────────
    if np.any(np.abs(beta_hat) > 50):
        warnings.warn(
            "Some |beta_hat| > 50. This often signals near-complete separation. "
            "Fitted probabilities near 0 or 1 should be treated with caution.",
            RuntimeWarning,
        )

    if not result.success:
        warnings.warn(
            f"Optimiser did not converge: {result.message}",
            RuntimeWarning,
        )

    probs   = expit(X @ beta_hat)
    nll_val = neg_log_likelihood(beta_hat, X, y)

    return dict(
        beta_hat = beta_hat,
        se       = se,
        probs    = probs,
        log_lik  = -nll_val,
        nll      = nll_val,
        status   = result.success,
        message  = result.message,
        result   = result,
        X        = X,
        d        = d,
        s        = s,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helper
# ─────────────────────────────────────────────────────────────────────────────

def plot_B(s_range, beta, phi_list, dphi_list=None, ax=None,
           label="B(s)", title="Discriminability function B(s)", **line_kw):
    """
    Plot B_beta(s) over s_range, and optionally B'(s) on a twin axis.

    Parameters
    ----------
    s_range   : array-like of s values
    beta      : coefficient vector
    phi_list  : basis functions
    dphi_list : derivative functions (optional; plotted on right axis if given)
    ax        : matplotlib Axes (created if None)
    label     : legend label for B(s) curve
    title     : plot title

    Returns
    -------
    ax : primary Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    s_arr = np.asarray(s_range, dtype=float)
    Bvals = B(s_arr, beta, phi_list)

    ax.plot(s_arr, Bvals, lw=2, label=label, **line_kw)
    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.4)
    ax.set_xlabel("s = r_i + r_j", fontsize=11)
    ax.set_ylabel("B(s)", fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.2)

    if dphi_list is not None:
        dBvals = B(s_arr, beta, dphi_list)
        ax2 = ax.twinx()
        ax2.plot(s_arr, dBvals, lw=1.5, ls=":", color="tomato", label="B'(s)")
        ax2.axhline(0, color="tomato", lw=0.6, ls="--", alpha=0.4)
        ax2.set_ylabel("B'(s)", color="tomato", fontsize=10)
        ax2.tick_params(axis="y", labelcolor="tomato")
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(fontsize=9)
        ax2.legend(lines2, labels2, fontsize=8, loc="upper right")
    else:
        ax.legend(fontsize=9)

    return ax


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic-data demo
# ─────────────────────────────────────────────────────────────────────────────

def synthetic_demo(seed=42, N=30, T=500, max_degree=1, constrained=False):
    """
    Fit the model on synthetic data with a known true B(s) = beta0 + beta1*s.

    True values:  beta = [3.0, -0.5]  →  B(s) = 3 - 0.5s  (positive, decreasing)

    Parameters
    ----------
    seed        : random seed
    N           : number of items
    T           : number of comparisons
    max_degree  : polynomial degree for basis (1 = linear in s)
    constrained : whether to also run constrained fit

    Returns
    -------
    out_unc, out_con (or None if not constrained)
    """
    rng = np.random.default_rng(seed)

    # Item ratings: Uniform(0.5, 2.5)
    r_true = rng.uniform(0.5, 2.5, size=N)

    # True beta
    beta_true = np.array([3.0, -0.5])

    # Random pairs
    idx = rng.integers(0, N, size=(T, 2))
    # Ensure i != j
    same = idx[:, 0] == idx[:, 1]
    idx[same, 1] = (idx[same, 1] + 1) % N

    r_i_idx, r_j_idx = idx[:, 0], idx[:, 1]

    phi_list, dphi_list = make_poly_basis(max_degree)

    # Compute true probabilities and sample y
    X_true, _, _ = build_design_matrix(r_i_idx, r_j_idx, r_true, phi_list)
    eta_true = X_true @ beta_true
    p_true   = expit(eta_true)
    y        = (rng.uniform(size=T) < p_true).astype(float)

    print(f"Synthetic demo: N={N} items, T={T} comparisons, basis degree={max_degree}")
    print(f"True beta: {beta_true}")
    print(f"Fraction y=1: {y.mean():.3f}")
    print()

    # Unconstrained fit
    out_unc = fit(r_i_idx, r_j_idx, y, r_true, phi_list, dphi_list=dphi_list)
    print("── Unconstrained fit ──────────────────────────────")
    _print_result(out_unc, beta_true)

    out_con = None
    if constrained:
        print()
        print("── Constrained fit (B>0, B'<=0) ───────────────────")
        out_con = fit(
            r_i_idx, r_j_idx, y, r_true, phi_list,
            dphi_list=dphi_list, constrained=True,
        )
        _print_result(out_con, beta_true)

    # Plot B(s)
    s_obs = r_true[r_i_idx] + r_true[r_j_idx]
    s_range = np.linspace(s_obs.min(), s_obs.max(), 200)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(s_range, B(s_range, beta_true, phi_list),
            "k--", lw=1.5, label="True B(s)")
    plot_B(s_range, out_unc["beta_hat"], phi_list, ax=ax,
           label=f"Fitted (unc.) B(s)", color="steelblue")
    if out_con is not None:
        plot_B(s_range, out_con["beta_hat"], phi_list, ax=ax,
               label=f"Fitted (con.) B(s)", color="tomato")
    ax.set_title("Synthetic demo: true vs fitted B(s)")
    ax.legend()
    plt.tight_layout()
    plt.show()

    return out_unc, out_con


def _print_result(out, beta_true=None):
    bhat = out["beta_hat"]
    se   = out["se"]
    K    = len(bhat)
    print(f"  log-likelihood: {out['log_lik']:.4f}")
    print(f"  converged:      {out['status']}  ({out['message']})")
    print(f"  {'k':>4}  {'beta_hat':>10}  {'SE':>8}  {'true':>8}")
    for k in range(K):
        true_str = f"{beta_true[k]:>8.4f}" if beta_true is not None and k < len(beta_true) else ""
        se_str   = f"{se[k]:>8.4f}"        if se is not None else "    n/a "
        print(f"  {k:>4}  {bhat[k]:>10.4f}  {se_str}  {true_str}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    synthetic_demo(constrained=True)
