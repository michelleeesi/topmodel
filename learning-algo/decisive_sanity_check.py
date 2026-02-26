#!/usr/bin/env python3
"""
Decisive-Only Sanity Check Experiment

Tests whether the oracle is consistent with a standard Bradley-Terry / logistic model
when τ = τ′ = 0, and whether any non-improvement comes from BT implementation constraints.

Experiments:
    A:  Random sampling + unconstrained logistic regression
    B1: Random sampling + simplex-constrained (scale s=1)
    B2: Random sampling + simplex-constrained (learnable scale s)
    C:  BALD selection + simplex-constrained (scale s=1)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable, Dict
from scipy.optimize import minimize
from scipy.special import expit as sigmoid
from collections import Counter
import warnings
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import os

warnings.filterwarnings('ignore')

# ============================================================================
# Constants
# ============================================================================
FEATURE_NAMES = ['elderlyDep', 'lifeYearsGained', 'obesity', 'weeklyWorkhours', 'yearsWaiting']
DIM = len(FEATURE_NAMES)

# ============================================================================
# Data Structures
# ============================================================================
@dataclass
class Patient:
    """Represents a patient with feature values."""
    elderlyDep: float
    lifeYearsGained: float
    obesity: float
    weeklyWorkhours: float
    yearsWaiting: float

    def to_array(self) -> np.ndarray:
        return np.array([
            self.elderlyDep, self.lifeYearsGained, self.obesity,
            self.weeklyWorkhours, self.yearsWaiting
        ], dtype=float)


@dataclass
class PairwiseQuery:
    """Represents a pairwise comparison query."""
    patient_left: Patient
    patient_right: Patient
    context: Optional[str] = None


# ============================================================================
# Helper Functions
# ============================================================================
def phi(query: PairwiseQuery) -> np.ndarray:
    """Feature difference vector: x_left - x_right."""
    return query.patient_left.to_array() - query.patient_right.to_array()


def generate_random_patient_normalized(rng: np.random.Generator) -> Patient:
    """Generate a random patient with features in [0, 1]."""
    return Patient(
        elderlyDep=rng.uniform(0, 1),
        lifeYearsGained=rng.uniform(0, 1),
        obesity=rng.uniform(0, 1),
        weeklyWorkhours=rng.uniform(0, 1),
        yearsWaiting=rng.uniform(0, 1),
    )


def generate_candidate_queries(
    n_candidates: int,
    rng: np.random.Generator,
) -> List[PairwiseQuery]:
    """Generate candidate queries with normalized features."""
    candidates = []
    for _ in range(n_candidates):
        left = generate_random_patient_normalized(rng)
        right = generate_random_patient_normalized(rng)
        candidates.append(PairwiseQuery(left, right))
    return candidates


def create_noise_fn(
    noise_type: str,
    scale_delta: float,
    scale_r: float,
    rng: np.random.Generator,
) -> Callable:
    """Create noise function for latent margins."""
    def noise_fn(delta: float, r: float) -> Tuple[float, float]:
        if noise_type == 'logistic':
            eps_delta = rng.logistic(0, scale_delta)
            eps_r = rng.logistic(0, scale_r) if scale_r > 0 else 0
        else:
            eps_delta = rng.normal(0, scale_delta)
            eps_r = rng.normal(0, scale_r) if scale_r > 0 else 0
        delta_tilde = delta + eps_delta
        r_tilde = r * np.exp(eps_r) if scale_r > 0 else r
        return delta_tilde, r_tilde
    return noise_fn


def compute_frame_gaps(
    query: PairwiseQuery,
    lambda_x: float = 1.0,
) -> Tuple[np.ndarray, set]:
    """Compute frame-level gaps and identify active frames."""
    feature_diff = phi(query)
    gaps = lambda_x * feature_diff
    active_frames = set(np.where(np.abs(gaps) > 0)[0].tolist())
    return gaps, active_frames


def compute_aggregate_scores(
    gaps: np.ndarray,
    weights: np.ndarray,
    active_frames: set
) -> Tuple[float, float]:
    """Compute aggregate preference score Δ(ω) and intensity r(ω)."""
    delta_omega = float(np.dot(gaps, weights))
    r_omega = float(sum(weights[j] * abs(gaps[j]) for j in active_frames))
    return delta_omega, r_omega


def predict_response_noisy(
    query: PairwiseQuery,
    weights: np.ndarray,
    noise_fn: Callable,
    tau: float,
    lambda_x: float,
    tau_prime: float,
) -> str:
    """
    Generate response from multi-frame model with noise.
    Returns: 'left', 'right', 'indifferent', or 'incomparable'
    """
    gaps, active_frames = compute_frame_gaps(query, lambda_x)
    delta, r = compute_aggregate_scores(gaps, weights, active_frames)
    delta_tilde, r_tilde = noise_fn(delta, r)

    if r_tilde < tau:
        return 'indifferent'
    if abs(delta_tilde) < tau_prime * r_tilde:
        return 'incomparable'
    return 'left' if delta_tilde >= tau_prime * r_tilde else 'right'


# ============================================================================
# BT Model Helpers (for simplex-constrained and BALD)
# ============================================================================
def project_to_simplex(v: np.ndarray) -> np.ndarray:
    """Project vector v onto the probability simplex."""
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.where(u > (cssv - 1) / np.arange(1, n + 1))[0]
    if len(rho) == 0:
        rho = n - 1
    else:
        rho = rho[-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    w = np.maximum(v - theta, 0)
    return w


def fit_bt_map_simplex(
    phis: np.ndarray,
    ys: np.ndarray,
    dim: int,
    l2_reg: float = 1e-2,
    n_restarts: int = 5,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Fit BT MAP estimate under simplex constraints (scale s=1)."""
    if rng is None:
        rng = np.random.default_rng()

    if len(phis) == 0:
        return np.ones(dim) / dim

    def neg_log_posterior(omega):
        logits = phis @ omega
        log_p = np.where(
            logits >= 0,
            -np.log1p(np.exp(-logits)),
            logits - np.log1p(np.exp(logits))
        )
        log_1_minus_p = np.where(
            logits >= 0,
            -logits - np.log1p(np.exp(-logits)),
            -np.log1p(np.exp(logits))
        )
        nll = -np.sum(ys * log_p + (1 - ys) * log_1_minus_p)
        reg = 0.5 * l2_reg * np.sum(omega ** 2)
        return nll + reg

    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    bounds = [(0, None) for _ in range(dim)]

    best_result = None
    for _ in range(n_restarts):
        omega0 = rng.dirichlet(np.ones(dim))
        result = minimize(
            neg_log_posterior, omega0, method='SLSQP',
            bounds=bounds, constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-8}
        )
        if best_result is None or result.fun < best_result.fun:
            best_result = result

    omega_hat = np.maximum(best_result.x, 0)
    omega_hat = omega_hat / omega_hat.sum()
    return omega_hat


def fit_bt_simplex_with_scale(
    phis: np.ndarray,
    ys: np.ndarray,
    dim: int,
    l2_reg: float = 1e-2,
    n_restarts: int = 5,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, float]:
    """
    Fit BT model with simplex-constrained weights AND learnable scale s.

    p(y=1|phi) = sigmoid(s * omega^T phi)

    Uses softmax parameterization for omega and softplus for s.

    Returns:
        omega: weights on simplex
        s: learned scale parameter
    """
    if rng is None:
        rng = np.random.default_rng()

    if len(phis) == 0:
        return np.ones(dim) / dim, 1.0

    def neg_log_likelihood(params):
        u = params[:dim]  # softmax params
        t = params[dim]   # softplus param for scale

        omega = np.exp(u - np.max(u))
        omega = omega / omega.sum()
        s = np.log1p(np.exp(t))  # softplus

        logits = s * (phis @ omega)
        log_p = np.where(
            logits >= 0,
            -np.log1p(np.exp(-logits)),
            logits - np.log1p(np.exp(logits))
        )
        log_1_minus_p = np.where(
            logits >= 0,
            -logits - np.log1p(np.exp(-logits)),
            -np.log1p(np.exp(logits))
        )
        nll = -np.sum(ys * log_p + (1 - ys) * log_1_minus_p)

        # Light regularization on u to prevent extreme softmax values
        reg = 0.5 * l2_reg * np.sum(u ** 2)
        return nll + reg

    best_result = None
    for _ in range(n_restarts):
        # Initialize u to get roughly uniform omega, t to get s near 1
        u0 = rng.normal(0, 0.1, size=dim)
        t0 = np.log(np.exp(1.0) - 1)  # inverse softplus of 1.0
        params0 = np.concatenate([u0, [t0]])

        result = minimize(
            neg_log_likelihood, params0, method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-8}
        )
        if best_result is None or result.fun < best_result.fun:
            best_result = result

    u_opt = best_result.x[:dim]
    t_opt = best_result.x[dim]

    omega = np.exp(u_opt - np.max(u_opt))
    omega = omega / omega.sum()
    s = np.log1p(np.exp(t_opt))

    return omega, s


def bt_laplace_covariance(
    phis: np.ndarray,
    omega_map: np.ndarray,
    l2_reg: float = 1e-2,
    jitter: float = 1e-6,
) -> np.ndarray:
    """Compute Laplace approximation covariance for BT posterior."""
    dim = len(omega_map)

    if len(phis) == 0:
        return np.eye(dim) / l2_reg

    logits = phis @ omega_map
    probs = sigmoid(logits)
    weights = probs * (1 - probs)
    H = (phis.T * weights) @ phis + l2_reg * np.eye(dim)
    H += jitter * np.eye(dim)

    try:
        sigma = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        sigma = np.linalg.pinv(H)

    return sigma


def sample_bt_laplace_posterior(
    omega_map: np.ndarray,
    sigma: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample from Laplace approximation and project to simplex."""
    raw_samples = rng.multivariate_normal(omega_map, sigma, size=n_samples)
    samples = np.array([project_to_simplex(s) for s in raw_samples])
    return samples


def bernoulli_entropy(p: np.ndarray) -> np.ndarray:
    """Compute binary entropy H(p) = -p*log(p) - (1-p)*log(1-p)."""
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -p * np.log(p) - (1 - p) * np.log(1 - p)


def bald_bernoulli_from_samples(
    phi_vec: np.ndarray,
    omega_samples: np.ndarray,
) -> float:
    """Compute BALD score for a query using BT posterior samples."""
    logits = omega_samples @ phi_vec
    probs = sigmoid(logits)

    mean_p = probs.mean()
    H_mean = bernoulli_entropy(np.array([mean_p]))[0]

    entropies = bernoulli_entropy(probs)
    mean_H = entropies.mean()

    return H_mean - mean_H


# ============================================================================
# Evaluation Helpers
# ============================================================================
def compute_log_loss(
    phis: np.ndarray,
    ys: np.ndarray,
    beta: np.ndarray,
    scale: float = 1.0
) -> float:
    """Compute log-loss on a dataset."""
    logits = scale * (phis @ beta)
    probs = sigmoid(logits)
    probs = np.clip(probs, 1e-15, 1 - 1e-15)
    ll = -np.mean(ys * np.log(probs) + (1 - ys) * np.log(1 - probs))
    return ll


def compute_accuracy(
    phis: np.ndarray,
    ys: np.ndarray,
    beta: np.ndarray,
    scale: float = 1.0
) -> float:
    """Compute accuracy on a dataset."""
    logits = scale * (phis @ beta)
    preds = (logits > 0).astype(float)
    return np.mean(preds == ys)


def generate_dataset(
    n_samples: int,
    oracle_weights: np.ndarray,
    tau: float,
    tau_prime: float,
    lambda_x: float,
    noise_type: str,
    scale_delta: float,
    scale_r: float,
    rng: np.random.Generator,
) -> Tuple[List[PairwiseQuery], np.ndarray, np.ndarray, Dict]:
    """
    Generate a dataset of queries and labels.

    Returns:
        queries: List of PairwiseQuery objects
        phis: Feature difference matrix (N, dim)
        ys: Labels (1 for left, 0 for right)
        stats: Dict with response type counts
    """
    noise_fn = create_noise_fn(noise_type, scale_delta, scale_r, rng)

    queries = []
    phis_list = []
    ys_list = []
    response_counts = Counter()

    for _ in range(n_samples):
        left = generate_random_patient_normalized(rng)
        right = generate_random_patient_normalized(rng)
        query = PairwiseQuery(left, right)

        response = predict_response_noisy(
            query, oracle_weights, noise_fn, tau, lambda_x, tau_prime
        )
        response_counts[response] += 1

        if response == 'left':
            queries.append(query)
            phis_list.append(phi(query))
            ys_list.append(1.0)
        elif response == 'right':
            queries.append(query)
            phis_list.append(phi(query))
            ys_list.append(0.0)
        # Skip indifferent/incomparable

    if len(phis_list) == 0:
        return [], np.zeros((0, len(oracle_weights))), np.zeros(0), response_counts

    return queries, np.array(phis_list), np.array(ys_list), response_counts


# ============================================================================
# Experiment A: Random Sampling + Unconstrained Logistic Regression
# ============================================================================
def run_experiment_A(
    k: int,
    oracle_weights: np.ndarray,
    test_phis: np.ndarray,
    test_ys: np.ndarray,
    tau: float,
    tau_prime: float,
    lambda_x: float,
    noise_type: str,
    scale_delta: float,
    scale_r: float,
    rng: np.random.Generator,
) -> Dict:
    """
    Experiment A: Random sampling with unconstrained logistic regression.

    Returns dict with:
        - logloss: test log-loss
        - accuracy: test accuracy
        - response_counts: dict of response type counts
    """
    # Generate training data
    _, train_phis, train_ys, response_counts = generate_dataset(
        n_samples=k,
        oracle_weights=oracle_weights,
        tau=tau,
        tau_prime=tau_prime,
        lambda_x=lambda_x,
        noise_type=noise_type,
        scale_delta=scale_delta,
        scale_r=scale_r,
        rng=rng,
    )

    # Check that we got decisive responses
    n_decisive = len(train_ys)
    if n_decisive < 2:
        return {
            'logloss': np.nan,
            'accuracy': np.nan,
            'response_counts': response_counts,
            'n_train': n_decisive,
        }

    # Fit unconstrained logistic regression (no intercept, weak regularization)
    lr = LogisticRegression(
        fit_intercept=False,  # BT model has no intercept
        C=100.0,  # weak L2 regularization (large C = small penalty)
        solver='lbfgs',
        max_iter=1000,
    )
    lr.fit(train_phis, train_ys)
    beta = lr.coef_.flatten()

    # Evaluate on test set
    logloss = compute_log_loss(test_phis, test_ys, beta, scale=1.0)
    accuracy = compute_accuracy(test_phis, test_ys, beta, scale=1.0)

    return {
        'logloss': logloss,
        'accuracy': accuracy,
        'response_counts': response_counts,
        'n_train': n_decisive,
        'beta': beta,
    }


# ============================================================================
# Experiment B1: Simplex-Constrained (s=1)
# ============================================================================
def run_experiment_B1(
    k: int,
    oracle_weights: np.ndarray,
    test_phis: np.ndarray,
    test_ys: np.ndarray,
    tau: float,
    tau_prime: float,
    lambda_x: float,
    noise_type: str,
    scale_delta: float,
    scale_r: float,
    rng: np.random.Generator,
) -> Dict:
    """
    Experiment B1: Random sampling with simplex-constrained weights, fixed s=1.
    """
    # Generate training data
    _, train_phis, train_ys, response_counts = generate_dataset(
        n_samples=k,
        oracle_weights=oracle_weights,
        tau=tau,
        tau_prime=tau_prime,
        lambda_x=lambda_x,
        noise_type=noise_type,
        scale_delta=scale_delta,
        scale_r=scale_r,
        rng=rng,
    )

    n_decisive = len(train_ys)
    if n_decisive < 2:
        return {
            'logloss': np.nan,
            'accuracy': np.nan,
            'response_counts': response_counts,
            'n_train': n_decisive,
        }

    dim = len(oracle_weights)
    omega = fit_bt_map_simplex(train_phis, train_ys, dim, l2_reg=1e-2, n_restarts=5, rng=rng)

    logloss = compute_log_loss(test_phis, test_ys, omega, scale=1.0)
    accuracy = compute_accuracy(test_phis, test_ys, omega, scale=1.0)

    return {
        'logloss': logloss,
        'accuracy': accuracy,
        'response_counts': response_counts,
        'n_train': n_decisive,
        'omega': omega,
    }


# ============================================================================
# Experiment B2: Simplex-Constrained with Learnable Scale
# ============================================================================
def run_experiment_B2(
    k: int,
    oracle_weights: np.ndarray,
    test_phis: np.ndarray,
    test_ys: np.ndarray,
    tau: float,
    tau_prime: float,
    lambda_x: float,
    noise_type: str,
    scale_delta: float,
    scale_r: float,
    rng: np.random.Generator,
) -> Dict:
    """
    Experiment B2: Random sampling with simplex-constrained weights AND learnable scale s.
    """
    # Generate training data
    _, train_phis, train_ys, response_counts = generate_dataset(
        n_samples=k,
        oracle_weights=oracle_weights,
        tau=tau,
        tau_prime=tau_prime,
        lambda_x=lambda_x,
        noise_type=noise_type,
        scale_delta=scale_delta,
        scale_r=scale_r,
        rng=rng,
    )

    n_decisive = len(train_ys)
    if n_decisive < 2:
        return {
            'logloss': np.nan,
            'accuracy': np.nan,
            'response_counts': response_counts,
            'n_train': n_decisive,
        }

    dim = len(oracle_weights)
    omega, s = fit_bt_simplex_with_scale(train_phis, train_ys, dim, l2_reg=1e-2, n_restarts=5, rng=rng)

    logloss = compute_log_loss(test_phis, test_ys, omega, scale=s)
    accuracy = compute_accuracy(test_phis, test_ys, omega, scale=s)

    return {
        'logloss': logloss,
        'accuracy': accuracy,
        'response_counts': response_counts,
        'n_train': n_decisive,
        'omega': omega,
        'scale': s,
    }


# ============================================================================
# Experiment C: BALD Selection with τ=τ′=0
# ============================================================================
def run_experiment_C(
    k: int,
    oracle_weights: np.ndarray,
    test_phis: np.ndarray,
    test_ys: np.ndarray,
    tau: float,
    tau_prime: float,
    lambda_x: float,
    noise_type: str,
    scale_delta: float,
    scale_r: float,
    n_candidates: int,
    n_posterior_samples: int,
    rng: np.random.Generator,
) -> Dict:
    """
    Experiment C: BALD-based query selection with simplex-constrained model.

    Run "until k" queries using BALD acquisition, then evaluate.
    """
    dim = len(oracle_weights)
    noise_fn = create_noise_fn(noise_type, scale_delta, scale_r, rng)

    phis_list = []
    ys_list = []
    response_counts = Counter()

    omega_map = np.ones(dim) / dim
    omega_samples = np.tile(omega_map, (n_posterior_samples, 1))

    for t in range(k):
        # Generate candidates
        candidates = generate_candidate_queries(n_candidates, rng)

        # Score with BALD
        best_query = None
        best_score = -np.inf
        for query in candidates:
            phi_vec = phi(query)
            score = bald_bernoulli_from_samples(phi_vec, omega_samples)
            if score > best_score:
                best_score = score
                best_query = query

        if best_query is None:
            best_query = candidates[0]

        # Query oracle
        response = predict_response_noisy(
            best_query, oracle_weights, noise_fn, tau, lambda_x, tau_prime
        )
        response_counts[response] += 1

        if response == 'left':
            phis_list.append(phi(best_query))
            ys_list.append(1.0)
        elif response == 'right':
            phis_list.append(phi(best_query))
            ys_list.append(0.0)
        # With τ=τ′=0, all should be decisive

        # Update posterior
        if len(phis_list) > 0:
            train_phis = np.array(phis_list)
            train_ys = np.array(ys_list)
            omega_map = fit_bt_map_simplex(train_phis, train_ys, dim, l2_reg=1e-2, n_restarts=3, rng=rng)
            sigma = bt_laplace_covariance(train_phis, omega_map, l2_reg=1e-2)
            omega_samples = sample_bt_laplace_posterior(omega_map, sigma, n_posterior_samples, rng)

    n_decisive = len(ys_list)
    if n_decisive < 2:
        return {
            'logloss': np.nan,
            'accuracy': np.nan,
            'response_counts': response_counts,
            'n_train': n_decisive,
        }

    logloss = compute_log_loss(test_phis, test_ys, omega_map, scale=1.0)
    accuracy = compute_accuracy(test_phis, test_ys, omega_map, scale=1.0)

    return {
        'logloss': logloss,
        'accuracy': accuracy,
        'response_counts': response_counts,
        'n_train': n_decisive,
        'omega': omega_map,
    }


# ============================================================================
# Main Experiment Runner
# ============================================================================
def run_decisive_sanity_check(
    K_values: List[int] = None,
    n_seeds: int = 20,
    n_test: int = 5000,
    oracle_weights: np.ndarray = None,
    tau: float = 0.0,
    tau_prime: float = 0.0,
    lambda_x: float = 1.0,
    noise_type: str = 'logistic',
    scale_delta: float = 0.1,
    scale_r: float = 0.0,
    n_candidates: int = 50,
    n_posterior_samples: int = 100,
    output_dir: str = '.',
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run the complete decisive sanity check experiment.

    Args:
        K_values: List of training set sizes to evaluate
        n_seeds: Number of random seeds per k
        n_test: Size of held-out test set
        oracle_weights: True weights (if None, uses uniform)
        tau: Indifference threshold (should be 0 for this experiment)
        tau_prime: Incomparability threshold (should be 0 for this experiment)
        lambda_x: Feature scaling
        noise_type: 'logistic' or 'normal'
        scale_delta: Noise scale for delta
        scale_r: Noise scale for r (0 means no r noise)
        n_candidates: Candidates per BALD round
        n_posterior_samples: Posterior samples for BALD
        output_dir: Directory to save outputs
        verbose: Print progress

    Returns:
        DataFrame with results
    """
    if K_values is None:
        K_values = [10, 20, 50, 100, 200, 500, 1000]

    if oracle_weights is None:
        oracle_weights = np.ones(DIM) / DIM

    # Verify tau and tau_prime are 0
    if tau != 0.0 or tau_prime != 0.0:
        print(f"WARNING: tau={tau}, tau_prime={tau_prime} are not 0!")
        print("This experiment is designed for tau=tau_prime=0 to ensure all responses are decisive.")

    # Master RNG
    master_rng = np.random.default_rng(42)

    # Generate fixed test set
    if verbose:
        print(f"Generating test set of size {n_test}...")
    test_rng = np.random.default_rng(master_rng.integers(0, 2**31))
    _, test_phis, test_ys, test_counts = generate_dataset(
        n_samples=n_test,
        oracle_weights=oracle_weights,
        tau=tau,
        tau_prime=tau_prime,
        lambda_x=lambda_x,
        noise_type=noise_type,
        scale_delta=scale_delta,
        scale_r=scale_r,
        rng=test_rng,
    )

    if verbose:
        print(f"Test set response counts: {dict(test_counts)}")
        print(f"Decisive test samples: {len(test_ys)}")
        n_non_decisive = test_counts.get('indifferent', 0) + test_counts.get('incomparable', 0)
        if n_non_decisive > 0:
            print(f"WARNING: {n_non_decisive} non-decisive responses in test set with tau=tau_prime={tau}!")

    # Store results
    results = []

    # Run experiments
    methods = ['A', 'B1', 'B2', 'C']

    for k in K_values:
        if verbose:
            print(f"\nRunning experiments for k={k}...")

        for seed in tqdm(range(n_seeds), desc=f'k={k}', disable=not verbose):
            seed_value = master_rng.integers(0, 2**31)

            for method in methods:
                rng = np.random.default_rng(seed_value)

                if method == 'A':
                    result = run_experiment_A(
                        k, oracle_weights, test_phis, test_ys,
                        tau, tau_prime, lambda_x,
                        noise_type, scale_delta, scale_r, rng
                    )
                elif method == 'B1':
                    result = run_experiment_B1(
                        k, oracle_weights, test_phis, test_ys,
                        tau, tau_prime, lambda_x,
                        noise_type, scale_delta, scale_r, rng
                    )
                elif method == 'B2':
                    result = run_experiment_B2(
                        k, oracle_weights, test_phis, test_ys,
                        tau, tau_prime, lambda_x,
                        noise_type, scale_delta, scale_r, rng
                    )
                elif method == 'C':
                    result = run_experiment_C(
                        k, oracle_weights, test_phis, test_ys,
                        tau, tau_prime, lambda_x,
                        noise_type, scale_delta, scale_r,
                        n_candidates, n_posterior_samples, rng
                    )

                results.append({
                    'seed': seed,
                    'k': k,
                    'method': method,
                    'logloss': result['logloss'],
                    'acc': result['accuracy'],
                    'n_train': result['n_train'],
                    'n_left': result['response_counts'].get('left', 0),
                    'n_right': result['response_counts'].get('right', 0),
                    'n_indiff': result['response_counts'].get('indifferent', 0),
                    'n_incomp': result['response_counts'].get('incomparable', 0),
                })

    df = pd.DataFrame(results)

    # Save CSV
    csv_path = os.path.join(output_dir, 'decisive_sanity_check_results.csv')
    df.to_csv(csv_path, index=False)
    if verbose:
        print(f"\nResults saved to {csv_path}")

    return df


def plot_results(
    df: pd.DataFrame,
    output_dir: str = '.',
    show: bool = True,
) -> None:
    """
    Create the main figure with 2 panels: log-loss vs k, accuracy vs k.
    """
    methods = ['A', 'B1', 'B2', 'C']
    method_labels = {
        'A': 'Unconstrained LR',
        'B1': 'Simplex (s=1)',
        'B2': 'Simplex (learnable s)',
        'C': 'BALD + Simplex',
    }
    colors = {
        'A': 'C0',
        'B1': 'C1',
        'B2': 'C2',
        'C': 'C3',
    }
    markers = {
        'A': 'o',
        'B1': 's',
        'B2': '^',
        'C': 'D',
    }

    K_values = sorted(df['k'].unique())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Log-loss
    ax1 = axes[0]
    for method in methods:
        method_df = df[df['method'] == method]
        means = []
        stderrs = []
        for k in K_values:
            k_data = method_df[method_df['k'] == k]['logloss'].dropna()
            means.append(k_data.mean())
            stderrs.append(k_data.std() / np.sqrt(len(k_data)) if len(k_data) > 1 else 0)

        ax1.errorbar(
            K_values, means, yerr=stderrs,
            marker=markers[method], color=colors[method],
            label=method_labels[method], capsize=3, linewidth=2, markersize=6
        )

    ax1.set_xlabel('Training Set Size (k)', fontsize=12)
    ax1.set_ylabel('Test Log-Loss', fontsize=12)
    ax1.set_title('Log-Loss vs Training Size', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)

    # Panel 2: Accuracy
    ax2 = axes[1]
    for method in methods:
        method_df = df[df['method'] == method]
        means = []
        stderrs = []
        for k in K_values:
            k_data = method_df[method_df['k'] == k]['acc'].dropna()
            means.append(k_data.mean())
            stderrs.append(k_data.std() / np.sqrt(len(k_data)) if len(k_data) > 1 else 0)

        ax2.errorbar(
            K_values, means, yerr=stderrs,
            marker=markers[method], color=colors[method],
            label=method_labels[method], capsize=3, linewidth=2, markersize=6
        )

    ax2.set_xlabel('Training Set Size (k)', fontsize=12)
    ax2.set_ylabel('Test Accuracy', fontsize=12)
    ax2.set_title('Accuracy vs Training Size', fontsize=14)
    ax2.legend(loc='lower right')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    fig_path = os.path.join(output_dir, 'decisive_sanity_check.pdf')
    plt.savefig(fig_path, bbox_inches='tight', dpi=150)
    print(f"Figure saved to {fig_path}")

    # Also save PNG
    png_path = os.path.join(output_dir, 'decisive_sanity_check.png')
    plt.savefig(png_path, bbox_inches='tight', dpi=150)
    print(f"Figure saved to {png_path}")

    if show:
        plt.show()
    else:
        plt.close()


def print_decisive_counts(df: pd.DataFrame) -> None:
    """Print summary of decisive vs non-decisive response counts."""
    print("\n" + "="*60)
    print("DECISIVE RESPONSE COUNTS")
    print("="*60)

    for method in df['method'].unique():
        method_df = df[df['method'] == method]
        total_left = method_df['n_left'].sum()
        total_right = method_df['n_right'].sum()
        total_indiff = method_df['n_indiff'].sum()
        total_incomp = method_df['n_incomp'].sum()
        total = total_left + total_right + total_indiff + total_incomp

        pct_decisive = 100 * (total_left + total_right) / total if total > 0 else 0

        print(f"\nMethod {method}:")
        print(f"  Left:         {total_left:6d} ({100*total_left/total:.1f}%)")
        print(f"  Right:        {total_right:6d} ({100*total_right/total:.1f}%)")
        print(f"  Indifferent:  {total_indiff:6d} ({100*total_indiff/total:.1f}%)")
        print(f"  Incomparable: {total_incomp:6d} ({100*total_incomp/total:.1f}%)")
        print(f"  DECISIVE:     {pct_decisive:.1f}%")


def interpret_results(df: pd.DataFrame) -> str:
    """Generate interpretation of the results."""
    K_values = sorted(df['k'].unique())
    k_small, k_large = K_values[0], K_values[-1]

    interpretation = []
    interpretation.append("\n" + "="*60)
    interpretation.append("INTERPRETATION")
    interpretation.append("="*60)

    # Check if unconstrained LR improves with k
    a_small = df[(df['method'] == 'A') & (df['k'] == k_small)]['logloss'].mean()
    a_large = df[(df['method'] == 'A') & (df['k'] == k_large)]['logloss'].mean()
    a_improves = a_large < a_small
    interpretation.append(f"\n1. Unconstrained LR (A): {'IMPROVES' if a_improves else 'DOES NOT IMPROVE'} with k")
    interpretation.append(f"   Log-loss at k={k_small}: {a_small:.4f}")
    interpretation.append(f"   Log-loss at k={k_large}: {a_large:.4f}")

    # Check if simplex constraint flattens
    b1_small = df[(df['method'] == 'B1') & (df['k'] == k_small)]['logloss'].mean()
    b1_large = df[(df['method'] == 'B1') & (df['k'] == k_large)]['logloss'].mean()
    b1_improves = b1_large < b1_small
    interpretation.append(f"\n2. Simplex s=1 (B1): {'IMPROVES' if b1_improves else 'FLATTENS/WORSENS'} with k")
    interpretation.append(f"   Log-loss at k={k_small}: {b1_small:.4f}")
    interpretation.append(f"   Log-loss at k={k_large}: {b1_large:.4f}")

    # Check if learnable scale helps
    b2_large = df[(df['method'] == 'B2') & (df['k'] == k_large)]['logloss'].mean()
    scale_helps = b2_large < b1_large
    interpretation.append(f"\n3. Learnable scale (B2): {'HELPS' if scale_helps else 'DOES NOT HELP'}")
    interpretation.append(f"   B1 log-loss at k={k_large}: {b1_large:.4f}")
    interpretation.append(f"   B2 log-loss at k={k_large}: {b2_large:.4f}")

    # Compare BALD to random
    c_large = df[(df['method'] == 'C') & (df['k'] == k_large)]['logloss'].mean()
    bald_better = c_large < b1_large
    interpretation.append(f"\n4. BALD vs Random (C vs B1): {'BALD BETTER' if bald_better else 'RANDOM BETTER'}")
    interpretation.append(f"   B1 (random) log-loss at k={k_large}: {b1_large:.4f}")
    interpretation.append(f"   C (BALD) log-loss at k={k_large}: {c_large:.4f}")

    # Overall conclusion
    interpretation.append("\n" + "-"*60)
    interpretation.append("OVERALL CONCLUSION:")
    if a_improves and not b1_improves:
        interpretation.append("  The simplex constraint appears to limit model expressiveness.")
        interpretation.append("  Unconstrained LR can fit the data better as k increases.")
    elif a_improves and b1_improves:
        interpretation.append("  Both methods improve with k, suggesting the oracle is")
        interpretation.append("  consistent with a BT model structure.")
    else:
        interpretation.append("  Need more data to draw conclusions.")

    return "\n".join(interpretation)


# ============================================================================
# Main
# ============================================================================
def main():
    """Run the decisive sanity check experiment."""
    print("="*60)
    print("DECISIVE-ONLY SANITY CHECK EXPERIMENT")
    print("="*60)
    print("\nThis experiment tests whether the oracle is consistent with")
    print("a standard Bradley-Terry model when τ = τ′ = 0.")
    print()

    # Configuration
    oracle_weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])  # Non-uniform for interesting dynamics
    oracle_weights = oracle_weights / oracle_weights.sum()  # Ensure on simplex

    print(f"Oracle weights: {oracle_weights}")
    print(f"Features: {FEATURE_NAMES}")
    print()

    # Run experiments
    df = run_decisive_sanity_check(
        K_values=[10, 20, 50, 100, 200, 500],
        n_seeds=20,
        n_test=5000,
        oracle_weights=oracle_weights,
        tau=0.0,
        tau_prime=0.0,
        lambda_x=1.0,
        noise_type='logistic',
        scale_delta=0.1,  # Small noise for clean signal
        scale_r=0.0,
        n_candidates=50,
        n_posterior_samples=100,
        output_dir='.',
        verbose=True,
    )

    # Print decisive counts
    print_decisive_counts(df)

    # Plot results
    plot_results(df, output_dir='.', show=False)

    # Print interpretation
    interpretation = interpret_results(df)
    print(interpretation)

    # Save interpretation
    with open('decisive_sanity_check_interpretation.txt', 'w') as f:
        f.write(interpretation)
    print("\nInterpretation saved to decisive_sanity_check_interpretation.txt")


if __name__ == '__main__':
    main()
