"""
Core classes and functions for active frame learning.

Shared between active_frame_learning_adaptive.ipynb, bradley_terry_simulation.ipynb,
and other notebooks.
"""

import numpy as np
from typing import List, Tuple, Dict, Set, Optional
from dataclasses import dataclass
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
from scipy.special import expit


# ============================================================================
# Configuration
# ============================================================================

FEATURE_NAMES = ['elderlyDep', 'lifeYearsGained', 'obesity', 'weeklyWorkhours', 'yearsWaiting']

FEATURE_RANGES = {
    'elderlyDep': (0, 5),
    'lifeYearsGained': (0, 25),
    'obesity': (0, 5),
    'weeklyWorkhours': (0, 50),
    'yearsWaiting': (1, 10)
}

TAU = 1.0           # Intensity threshold
TAU_PRIME = 0.2     # Resolvability threshold
LAMBDA_X = 1.0      # Query scaling factor


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class Patient:
    """Represents a patient with feature values."""
    elderlyDep: int
    lifeYearsGained: float
    obesity: int
    weeklyWorkhours: int
    yearsWaiting: int

    def to_array(self) -> np.ndarray:
        """Convert to numpy array in standard feature order."""
        return np.array([
            self.elderlyDep,
            self.lifeYearsGained,
            self.obesity,
            self.weeklyWorkhours,
            self.yearsWaiting
        ], dtype=float)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'Patient':
        """Create Patient from numpy array."""
        return cls(
            elderlyDep=int(arr[0]),
            lifeYearsGained=float(arr[1]),
            obesity=int(arr[2]),
            weeklyWorkhours=int(arr[3]),
            yearsWaiting=int(arr[4])
        )

    def __repr__(self):
        return f"Patient(elder={self.elderlyDep}, life={self.lifeYearsGained}, " \
               f"obesity={self.obesity}, work={self.weeklyWorkhours}, wait={self.yearsWaiting})"


@dataclass
class PairwiseQuery:
    """Represents a pairwise comparison query."""
    patient_left: Patient
    patient_right: Patient
    context: Optional[str] = None

    def __repr__(self):
        return f"Query:\n  LEFT:  {self.patient_left}\n  RIGHT: {self.patient_right}"


@dataclass
class QueryResponse:
    """Response to a pairwise query."""
    choice: str  # 'left', 'right', 'indifferent', 'incomparable'
    active_frames: Set[int]

    def __repr__(self):
        return f"Response(choice={self.choice}, active_frames={self.active_frames})"


# ============================================================================
# Simplex sampling
# ============================================================================

def sample_from_simplex(n_samples: int, dim: int,
                        active_frames: Optional[Set[int]] = None,
                        random_state: Optional[int] = None) -> np.ndarray:
    """
    Sample uniformly from the probability simplex using Dirichlet(1,...,1).

    Parameters
    ----------
    n_samples : int
    dim : int
    active_frames : Set[int], optional
        If provided, only these frames can have non-zero weight.
    random_state : int, optional

    Returns
    -------
    samples : np.ndarray, shape (n_samples, dim)
    """
    if random_state is not None:
        np.random.seed(random_state)

    if active_frames is None:
        samples = np.random.dirichlet(np.ones(dim), size=n_samples)
    else:
        samples = np.zeros((n_samples, dim))
        active_list = sorted(list(active_frames))
        n_active = len(active_list)
        if n_active == 0:
            raise ValueError("Must have at least one active frame")
        sub_samples = np.random.dirichlet(np.ones(n_active), size=n_samples)
        for i, frame_idx in enumerate(active_list):
            samples[:, frame_idx] = sub_samples[:, i]

    return samples


def resample_from_feasible_set(existing_samples: np.ndarray,
                                target_n_samples: int,
                                noise_scale: float = 0.05) -> np.ndarray:
    """
    Replenish feasible set by perturbing existing samples with Gaussian noise
    and re-projecting onto the simplex.
    """
    n_existing, dim = existing_samples.shape
    indices = np.random.choice(n_existing, size=target_n_samples, replace=True)
    new_samples = existing_samples[indices].copy()

    noise = np.random.randn(target_n_samples, dim) * noise_scale
    new_samples = new_samples + noise
    new_samples = np.maximum(new_samples, 0)

    row_sums = new_samples.sum(axis=1, keepdims=True)
    new_samples = new_samples / np.maximum(row_sums, 1e-10)

    return new_samples


# ============================================================================
# Convergence metrics
# ============================================================================

def compute_stable_convergence_metrics(samples: np.ndarray) -> dict:
    """Compute variance, average distance, and median distance."""
    if len(samples) == 0:
        return {'variance': float('inf'), 'avg_distance': float('inf'), 'median_distance': float('inf')}

    variance = np.sum(np.var(samples, axis=0))

    n_pairs = min(1000, len(samples) * (len(samples) - 1) // 2)
    if len(samples) > 50:
        idx1 = np.random.choice(len(samples), n_pairs)
        idx2 = np.random.choice(len(samples), n_pairs)
        distances = np.linalg.norm(samples[idx1] - samples[idx2], axis=1)
    else:
        distances = pdist(samples)

    return {
        'variance': variance,
        'avg_distance': np.mean(distances),
        'median_distance': np.median(distances)
    }


def compute_diameter(samples: np.ndarray, norm: str = 'l1') -> float:
    """Compute maximum pairwise distance (diameter) of sample set."""
    if len(samples) <= 1:
        return 0.0

    if norm == 'l1':
        p = 1
    elif norm == 'l2':
        p = 2
    elif norm == 'linf':
        p = np.inf
    else:
        raise ValueError(f"Unknown norm: {norm}")

    distances = pdist(samples, metric='minkowski', p=p)
    return distances.max()


# ============================================================================
# Query generation helpers
# ============================================================================

def generate_random_patient() -> Patient:
    """Generate a random patient with features in valid ranges."""
    return Patient(
        elderlyDep=np.random.randint(0, 5),
        lifeYearsGained=np.random.randint(0, 25),
        obesity=np.random.randint(0, 5),
        weeklyWorkhours=np.random.randint(0, 50),
        yearsWaiting=np.random.randint(1, 10)
    )


# ============================================================================
# Core computations
# ============================================================================

def compute_frame_gaps(query: PairwiseQuery,
                       lambda_x: float = LAMBDA_X,
                       tau: float = TAU) -> Tuple[np.ndarray, Set[int]]:
    """
    Compute frame-level gaps and identify active (decisive) frames.

    Returns
    -------
    gaps : np.ndarray, shape (n_frames,)
    active_frames : Set[int]
        Frames where |gap_j| >= tau.
    """
    left_features = query.patient_left.to_array()
    right_features = query.patient_right.to_array()
    gaps = lambda_x * (left_features - right_features)
    active_frames = set(np.where(np.abs(gaps) > tau)[0].tolist())
    return gaps, active_frames


def compute_aggregate_scores(gaps: np.ndarray,
                              weights: np.ndarray,
                              active_frames: Set[int]) -> Tuple[float, float]:
    """
    Compute aggregate preference score delta(omega) and intensity r(omega).
    """
    if len(active_frames) == 0:
        return 0.0, 0.0

    active_list = sorted(list(active_frames))
    active_gaps = gaps[active_list]
    active_weights = weights[active_list]

    delta_omega = np.dot(active_weights, active_gaps)
    r_omega = np.dot(active_weights, np.abs(active_gaps))

    return delta_omega, r_omega


def predict_response(query: PairwiseQuery,
                     weights: np.ndarray,
                     tau: float = TAU,
                     lambda_x: float = LAMBDA_X,
                     tau_prime: float = TAU_PRIME) -> str:
    """
    Predict response for a query given a weight vector.

    Returns one of: 'left', 'right', 'indifferent', 'incomparable'
    """
    gaps, active_frames = compute_frame_gaps(query, lambda_x, tau)
    delta_omega, r_omega = compute_aggregate_scores(gaps, weights, active_frames)

    if r_omega < tau:
        return 'indifferent'
    if r_omega >= tau and np.abs(delta_omega) < tau_prime * r_omega:
        return 'incomparable'
    elif r_omega >= tau and delta_omega >= tau_prime * r_omega:
        return 'left'
    elif r_omega >= tau and delta_omega <= -tau_prime * r_omega:
        return 'right'
    else:
        return 'indifferent'


# ============================================================================
# Volume removal
# ============================================================================

def filter_samples_by_response(samples: np.ndarray,
                                query: PairwiseQuery,
                                observed_response: str,
                                tau: float = TAU,
                                lambda_x: float = LAMBDA_X,
                                tau_prime: float = TAU_PRIME) -> np.ndarray:
    """
    Filter samples to keep only those consistent with observed response.
    """
    gaps, active_frames = compute_frame_gaps(query, lambda_x, tau)

    if len(active_frames) == 0:
        if observed_response == 'incomparable':
            return samples
        else:
            return np.empty((0, samples.shape[1]))

    active_list = sorted(list(active_frames))
    active_gaps = gaps[active_list]

    active_weights = samples[:, active_list]
    delta_omegas = np.dot(active_weights, active_gaps)
    r_omegas = np.dot(active_weights, np.abs(active_gaps))

    predicted = np.empty(len(samples), dtype=object)

    for idx in range(len(samples)):
        r_omega = r_omegas[idx]
        delta_omega = delta_omegas[idx]

        if r_omega < tau:
            predicted[idx] = 'indifferent'
        elif r_omega >= tau and np.abs(delta_omega) < tau_prime * r_omega:
            predicted[idx] = 'incomparable'
        elif r_omega >= tau and delta_omega >= tau_prime * r_omega:
            predicted[idx] = 'left'
        elif r_omega >= tau and delta_omega <= -tau_prime * r_omega:
            predicted[idx] = 'right'
        else:
            predicted[idx] = 'indifferent'

    consistent_mask = (predicted == observed_response)
    return samples[consistent_mask]


# ============================================================================
# Query generation
# ============================================================================

def generate_query_activating_frames(target_frames: Set[int],
                                      lambda_x: float = LAMBDA_X,
                                      tau: float = TAU,
                                      min_gap: float = None,
                                      max_attempts: int = 100) -> Optional[PairwiseQuery]:
    """Generate a query that activates a specific subset of frames."""
    if min_gap is None:
        min_gap = 1.5 * tau

    n_features = len(FEATURE_NAMES)

    for attempt in range(max_attempts):
        left = generate_random_patient()
        left_arr = left.to_array()
        right_arr = left_arr.copy()

        for j in range(n_features):
            feature_name = FEATURE_NAMES[j]
            min_val, max_val = FEATURE_RANGES[feature_name]

            if j in target_frames:
                required_diff = min_gap / lambda_x
                if left_arr[j] + required_diff <= max_val:
                    right_arr[j] = left_arr[j] - required_diff
                elif left_arr[j] - required_diff >= min_val:
                    right_arr[j] = left_arr[j] + required_diff
                else:
                    break
            else:
                max_diff = tau / lambda_x * 0.8
                diff = np.random.uniform(-max_diff, max_diff)
                right_arr[j] = np.clip(left_arr[j] + diff, min_val, max_val)
        else:
            right = Patient.from_array(right_arr)
            query = PairwiseQuery(left, right)
            _, active = compute_frame_gaps(query, lambda_x, tau)
            if active == target_frames:
                return query

    return None


def generate_candidate_queries(n_candidates: int = 50,
                                n_features: int = 5,
                                max_active: int = None) -> List[PairwiseQuery]:
    """Generate a diverse pool of candidate queries."""
    if max_active is None:
        max_active = n_features

    candidates = []
    attempted_patterns = set()
    max_attempts = n_candidates * 3

    for _ in range(max_attempts):
        if len(candidates) >= n_candidates:
            break

        if np.random.random() < 0.7:
            n_active = np.random.randint(2, min(5, max_active + 1))
        else:
            n_active = np.random.randint(1, max_active + 1)

        target_frames = set(np.random.choice(n_features, size=n_active, replace=False).tolist())

        pattern_key = frozenset(target_frames)
        if pattern_key in attempted_patterns:
            continue
        attempted_patterns.add(pattern_key)

        query = generate_query_activating_frames(target_frames)
        if query is not None:
            candidates.append(query)

    while len(candidates) < n_candidates * 0.5:
        left = generate_random_patient()
        right = generate_random_patient()
        candidates.append(PairwiseQuery(left, right))

    return candidates


# ============================================================================
# Query selection
# ============================================================================

def select_query_by_uncertainty(candidates: List[PairwiseQuery],
                                 samples: np.ndarray,
                                 tau: float = TAU,
                                 lambda_x: float = LAMBDA_X,
                                 tau_prime: float = TAU_PRIME) -> Tuple[PairwiseQuery, Dict[str, float]]:
    """Select the query that maximizes uncertainty (entropy of response distribution)."""
    if len(candidates) == 0:
        raise ValueError("No candidate queries provided")

    if len(samples) == 0:
        return candidates[0], {'uncertainty': 0.0}

    best_query = None
    best_uncertainty = -1
    best_info = {}

    for query in candidates:
        gaps, active_frames = compute_frame_gaps(query, lambda_x, tau)

        if len(active_frames) == 0:
            continue

        active_list = sorted(list(active_frames))
        active_gaps = gaps[active_list]
        active_weights = samples[:, active_list]

        delta_omegas = np.dot(active_weights, active_gaps)
        r_omegas = np.dot(active_weights, np.abs(active_gaps))

        response_counts = {'left': 0, 'right': 0, 'indifferent': 0, 'incomparable': 0}

        for idx in range(len(samples)):
            r_omega = r_omegas[idx]
            delta_omega = delta_omegas[idx]

            if r_omega < tau:
                response_counts['indifferent'] += 1
            elif r_omega >= tau and np.abs(delta_omega) < tau_prime * r_omega:
                response_counts['incomparable'] += 1
            elif r_omega >= tau and delta_omega >= tau_prime * r_omega:
                response_counts['left'] += 1
            elif r_omega >= tau and delta_omega <= -tau_prime * r_omega:
                response_counts['right'] += 1
            else:
                response_counts['indifferent'] += 1

        total = len(samples)
        probs = [count / total for count in response_counts.values() if count > 0]

        if len(probs) <= 1:
            uncertainty = 0.0
        else:
            uncertainty = -sum(p * np.log2(p) for p in probs)

        if uncertainty > best_uncertainty:
            best_uncertainty = uncertainty
            best_query = query
            best_info = {
                'uncertainty': uncertainty,
                'response_counts': response_counts.copy(),
                'active_frames': active_frames
            }

    if best_query is None:
        best_query = candidates[0]
        best_info = {'uncertainty': 0.0}

    return best_query, best_info


# ============================================================================
# Adaptive learning helpers
# ============================================================================

def compute_frame_uncertainties(samples: np.ndarray) -> np.ndarray:
    """Compute variance of each frame's weight across samples."""
    if len(samples) == 0:
        return np.zeros(samples.shape[1] if len(samples.shape) > 1 else 0)
    return np.var(samples, axis=0)


def generate_adaptive_candidates(samples: np.ndarray,
                                  n_candidates: int = 50,
                                  n_features: int = 5,
                                  top_k_uncertain: int = 3) -> List[PairwiseQuery]:
    """Generate candidate queries targeting frames with highest weight uncertainty."""
    uncertainties = compute_frame_uncertainties(samples)
    uncertain_frame_indices = np.argsort(uncertainties)[::-1][:top_k_uncertain]
    uncertain_frames = set(uncertain_frame_indices.tolist())

    candidates = []
    attempted_patterns = set()
    max_attempts = n_candidates * 3

    for _ in range(max_attempts):
        if len(candidates) >= n_candidates:
            break

        if np.random.random() < 0.8 and len(uncertain_frames) >= 2:
            n_active_uncertain = np.random.randint(2, min(4, len(uncertain_frames) + 1))
            selected_uncertain = set(np.random.choice(
                list(uncertain_frames),
                size=min(n_active_uncertain, len(uncertain_frames)),
                replace=False
            ).tolist())

            if np.random.random() < 0.5:
                other_frames = set(range(n_features)) - uncertain_frames
                if len(other_frames) > 0:
                    n_other = np.random.randint(0, min(3, len(other_frames) + 1))
                    if n_other > 0:
                        selected_other = set(np.random.choice(
                            list(other_frames),
                            size=n_other,
                            replace=False
                        ).tolist())
                        target_frames = selected_uncertain | selected_other
                    else:
                        target_frames = selected_uncertain
                else:
                    target_frames = selected_uncertain
            else:
                target_frames = selected_uncertain
        else:
            n_active = np.random.randint(2, min(5, n_features + 1))
            target_frames = set(np.random.choice(n_features, size=n_active, replace=False).tolist())

        pattern_key = frozenset(target_frames)
        if pattern_key in attempted_patterns:
            continue
        attempted_patterns.add(pattern_key)

        query = generate_query_activating_frames(target_frames)
        if query is not None:
            candidates.append(query)

    while len(candidates) < n_candidates * 0.5:
        left = generate_random_patient()
        right = generate_random_patient()
        candidates.append(PairwiseQuery(left, right))

    return candidates


def check_epsilon_pareto(samples: np.ndarray,
                          epsilon: float,
                          n_test_rules: int = 100) -> Tuple[bool, float]:
    """Check if feasible set diameter is within epsilon (epsilon-Pareto optimal)."""
    if len(samples) <= 1:
        return True, 0.0

    n_samples = len(samples)
    if n_samples <= n_test_rules:
        distances = pdist(samples, metric='cityblock')
        max_distance = distances.max() if len(distances) > 0 else 0.0
    else:
        max_distance = 0.0
        for _ in range(n_test_rules):
            idx1, idx2 = np.random.choice(n_samples, size=2, replace=False)
            distance = np.abs(samples[idx1] - samples[idx2]).sum()
            max_distance = max(max_distance, distance)

    return max_distance <= epsilon, max_distance


# ============================================================================
# Main learning loops
# ============================================================================

def active_learning_loop(n_initial_samples: int = 5000,
                          convergence_diameter: float = 0.1,
                          max_iterations: int = 100,
                          n_candidates: int = 50,
                          min_samples: int = 100,
                          resample_threshold: int = 50,
                          oracle_weights: Optional[np.ndarray] = None,
                          verbose: bool = True) -> Tuple[np.ndarray, List[Dict]]:
    """
    Baseline active learning algorithm using random candidate generation
    and diameter-based stopping.
    """
    n_features = len(FEATURE_NAMES)
    samples = sample_from_simplex(n_initial_samples, n_features, random_state=42)
    history = []

    if verbose:
        print(f"Active Learning for Frame Weights")
        print(f"{'='*60}")
        print(f"Initial samples: {len(samples)}")
        print(f"Convergence threshold: {convergence_diameter} (L1 diameter)")
        print(f"Max iterations: {max_iterations}\n")

    for iteration in range(max_iterations):
        diameter = compute_diameter(samples, norm='l1')

        if verbose:
            print(f"Iteration {iteration + 1}")
            print(f"  Feasible samples: {len(samples)}")
            print(f"  Diameter: {diameter:.4f}")

        if diameter <= convergence_diameter:
            if verbose:
                print(f"\n✓ Converged! Diameter {diameter:.4f} ≤ {convergence_diameter}")
            break

        candidates = generate_candidate_queries(n_candidates=n_candidates)
        if len(candidates) == 0:
            if verbose:
                print("  Warning: No valid candidates generated")
            break

        query, query_info = select_query_by_uncertainty(candidates, samples)

        if verbose:
            print(f"  Query uncertainty: {query_info.get('uncertainty', 0):.3f} bits")
            print(f"  Active frames: {sorted(query_info.get('active_frames', set()))}")

        if oracle_weights is not None:
            response = predict_response(query, oracle_weights)
        else:
            print(f"\n{query}")
            print("Please respond: left/right/indifferent/incomparable")
            response = input("Response: ").strip().lower()

        if verbose:
            print(f"  Response: {response}")

        samples_before = len(samples)
        samples = filter_samples_by_response(samples, query, response)
        samples_after = len(samples)

        if verbose:
            print(f"  Filtered: {samples_before} → {samples_after} samples")

        if samples_after < resample_threshold and samples_after > 0:
            samples = resample_from_feasible_set(samples, min_samples)
            if verbose:
                print(f"  Resampled to {len(samples)} samples")
        elif samples_after == 0:
            if verbose:
                print("  ERROR: No consistent samples remain!")
            break

        history.append({
            'iteration': iteration + 1,
            'query': query,
            'response': response,
            'diameter': diameter,
            'n_samples': samples_after,
            'uncertainty': query_info.get('uncertainty', 0),
            'active_frames': query_info.get('active_frames', set())
        })

        if verbose:
            print()

    learned_weights = samples.mean(axis=0)

    if verbose:
        print(f"\nLearned weights:")
        for i, (name, weight) in enumerate(zip(FEATURE_NAMES, learned_weights)):
            print(f"  {name:20s}: {weight:.4f}")
        if oracle_weights is not None:
            print(f"\nGround truth weights:")
            for i, (name, weight) in enumerate(zip(FEATURE_NAMES, oracle_weights)):
                print(f"  {name:20s}: {weight:.4f}")
            l1_error = np.abs(learned_weights - oracle_weights).sum()
            print(f"\nL1 error: {l1_error:.4f}")

    return learned_weights, history


def active_learning_loop_adaptive(n_initial_samples: int = 20000,
                                   epsilon_pareto: float = 0.15,
                                   max_iterations: int = 100,
                                   n_candidates: int = 50,
                                   top_k_uncertain: int = 3,
                                   min_samples: int = 100,
                                   resample_threshold: int = 0,
                                   oracle_weights: Optional[np.ndarray] = None,
                                   verbose: bool = True) -> Tuple[np.ndarray, List[Dict]]:
    """
    Adaptive active learning using uncertainty-targeted candidate generation
    and epsilon-Pareto stopping.
    """
    n_features = len(FEATURE_NAMES)
    samples = sample_from_simplex(n_initial_samples, n_features, random_state=42)
    history = []

    if verbose:
        print(f"ADAPTIVE Active Learning for Frame Weights")
        print(f"{'='*60}")
        print(f"Initial samples: {len(samples)}")
        print(f"Stopping criterion: ε-Pareto with ε = {epsilon_pareto}")
        print(f"Max iterations: {max_iterations}")
        print(f"Adaptive targeting: Top-{top_k_uncertain} uncertain frames\n")

    for iteration in range(max_iterations):
        is_pareto, max_distance = check_epsilon_pareto(samples, epsilon_pareto)

        if verbose:
            print(f"Iteration {iteration + 1}")
            print(f"  Feasible samples: {len(samples)}")
            print(f"  Max L1 distance: {max_distance:.4f}")
            print(f"  ε-Pareto (ε={epsilon_pareto}): {'✓ YES' if is_pareto else '✗ NO'}")

        if is_pareto:
            if verbose:
                print(f"\n✓ Converged! All rules are {epsilon_pareto}-Pareto optimal")
                print(f"   (Max L1 distance {max_distance:.4f} ≤ {epsilon_pareto})")
            break

        uncertainties = compute_frame_uncertainties(samples)
        if verbose:
            top_uncertain_indices = np.argsort(uncertainties)[::-1][:top_k_uncertain]
            top_uncertain_names = [FEATURE_NAMES[i] for i in top_uncertain_indices]
            print(f"  Most uncertain frames: {top_uncertain_names}")

        candidates = generate_adaptive_candidates(
            samples,
            n_candidates=n_candidates,
            top_k_uncertain=top_k_uncertain
        )

        if len(candidates) == 0:
            if verbose:
                print("  Warning: No valid candidates generated")
            break

        query, query_info = select_query_by_uncertainty(candidates, samples)

        if verbose:
            print(f"  Query uncertainty: {query_info.get('uncertainty', 0):.3f} bits")
            print(f"  Active frames: {sorted(query_info.get('active_frames', set()))}")

        if oracle_weights is not None:
            response = predict_response(query, oracle_weights)
        else:
            print(f"\n{query}")
            print("Please respond: left/right/indifferent/incomparable")
            response = input("Response: ").strip().lower()

        if verbose:
            print(f"  Response: {response}")

        samples_before = len(samples)
        samples = filter_samples_by_response(samples, query, response)
        samples_after = len(samples)

        if verbose:
            print(f"  Filtered: {samples_before} → {samples_after} samples")

        if samples_after < resample_threshold and samples_after > 0:
            samples = resample_from_feasible_set(samples, min_samples)
            if verbose:
                print(f"  Resampled to {len(samples)} samples")
        elif samples_after == 0:
            if verbose:
                print("  ERROR: No consistent samples remain!")
            break

        history.append({
            'iteration': iteration + 1,
            'query': query,
            'response': response,
            'max_distance': max_distance,
            'is_epsilon_pareto': is_pareto,
            'n_samples': samples_after,
            'uncertainty': query_info.get('uncertainty', 0),
            'active_frames': query_info.get('active_frames', set()),
            'frame_uncertainties': uncertainties.copy()
        })

        if verbose:
            print()

    learned_weights = samples.mean(axis=0)

    if verbose:
        print(f"\nLearned weights:")
        for i, (name, weight) in enumerate(zip(FEATURE_NAMES, learned_weights)):
            print(f"  {name:20s}: {weight:.4f}")
        if oracle_weights is not None:
            print(f"\nGround truth weights:")
            for i, (name, weight) in enumerate(zip(FEATURE_NAMES, oracle_weights)):
                print(f"  {name:20s}: {weight:.4f}")
            l1_error = np.abs(learned_weights - oracle_weights).sum()
            print(f"\nL1 error: {l1_error:.4f}")

    return learned_weights, history


# ============================================================================
# Bradley-Terry comparison utilities
# ============================================================================

def extract_transcript(history: List[Dict]) -> List[Tuple[PairwiseQuery, str]]:
    """Extract (query, response) pairs from learning history."""
    return [(h['query'], h['response']) for h in history]


def fit_bradley_terry(transcript: List[Tuple[PairwiseQuery, str]],
                       n_features: int = 5,
                       feature_names: List[str] = FEATURE_NAMES) -> np.ndarray:
    """
    Fit Bradley-Terry model via logistic regression on feature differences.
    Returns weights normalized to simplex.
    """
    X = []
    y = []

    for query, response in transcript:
        left_features = query.patient_left.to_array()
        right_features = query.patient_right.to_array()

        if response == 'left':
            X.append(left_features - right_features)
            y.append(1)
        elif response == 'right':
            X.append(left_features - right_features)
            y.append(0)

    if len(X) == 0:
        print("Warning: No decisive comparisons (left/right) in transcript!")
        return np.ones(n_features) / n_features

    X = np.array(X)
    y = np.array(y)

    print(f"Bradley-Terry fitting on {len(y)} decisive comparisons (out of {len(transcript)} total)")
    print(f"  Left preferred: {np.sum(y)} ({100*np.mean(y):.1f}%)")
    print(f"  Right preferred: {len(y) - np.sum(y)} ({100*(1-np.mean(y)):.1f}%)")

    def neg_log_likelihood(w):
        logits = X @ w
        loss = -np.mean(y * np.log(expit(logits) + 1e-10) +
                       (1 - y) * np.log(1 - expit(logits) + 1e-10))
        loss += 0.01 * np.sum(w**2)
        return loss

    w_init = np.ones(n_features) / n_features
    result = minimize(neg_log_likelihood, w_init, method='L-BFGS-B')

    if not result.success:
        print(f"Warning: Optimization did not converge: {result.message}")

    weights = np.maximum(result.x, 0)
    weights = weights / (np.sum(weights) + 1e-10)

    return weights
