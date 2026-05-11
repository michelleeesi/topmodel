# ============================================================================
# SKETCH: BT trial runner using hit-and-run MCMC (no Laplace approximation)
# ============================================================================
# Apples-to-apples counterpart to run_multiframe_trial: same posterior sampler
# (hit-and-run on the simplex), same BALD estimator structure (entropy over
# MCMC samples), same point estimate (sample mean). The ONLY thing that
# differs from run_multiframe_trial is the likelihood — 2-outcome BT instead
# of 4-outcome multi-frame.
#
# Drop this in as a notebook cell after the existing BT cell. It depends on:
#   - hit_and_run_simplex_step       (cell 4)
#   - generate_candidate_queries     (cell 1/2)
#   - phi, predict_response_noisy    (cell 2)
#   - bald_bernoulli_from_samples    (cell 7)
#   - bt_log_loss                    (cell 7 / utils)
#   - cosine_similarity, l1_error    (cell 1)
# ============================================================================


# ----------------------------------------------------------------------------
# 1. BT log-likelihood given a transcript
# ----------------------------------------------------------------------------
# BT only models left/right. Drop indecisive outcomes (or treat them via
# forced_choice, just like the Laplace runner does — see step 4 below).
def compute_bt_log_likelihood(
    phis_decisive: np.ndarray,   # (n_decisive, dim) feature differences
    ys_decisive: np.ndarray,     # (n_decisive,) {0, 1}
    omega: np.ndarray,           # (dim,) point on simplex
    scale: float = 1.0,
) -> float:
    """log p(y | omega, s) = sum log sigmoid((2y-1) * s * omega^T phi)."""
    if len(phis_decisive) == 0:
        return 0.0
    logits = scale * (phis_decisive @ omega)
    # numerically stable log-sigmoid
    log_p   = np.where(logits >= 0,
                       -np.log1p(np.exp(-logits)),
                       logits - np.log1p(np.exp(logits)))
    log_1mp = np.where(logits >= 0,
                       -logits - np.log1p(np.exp(-logits)),
                       -np.log1p(np.exp(logits)))
    return float(np.sum(ys_decisive * log_p + (1 - ys_decisive) * log_1mp))


# ----------------------------------------------------------------------------
# 2. Hit-and-run posterior sampler for BT (mirrors sample_posterior_hit_and_run)
# ----------------------------------------------------------------------------
# Two variants: fixed scale, or jointly sample (omega, scale) when learn_scale.
# The fixed-scale version is the apples-to-apples one — multi-frame doesn't
# learn its noise scale either when noise is "known".
def sample_bt_posterior_hit_and_run(
    phis_decisive: np.ndarray,
    ys_decisive: np.ndarray,
    n_samples: int = 300,
    burn_in: int = 200,
    scale: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, float]:
    """Hit-and-run on the simplex with the BT likelihood. Same shape as
    sample_posterior_hit_and_run — uniform Dirichlet prior, MH with hit-and-run
    proposals."""
    if rng is None:
        rng = np.random.default_rng()
    dim = DIM
    omega = np.ones(dim) / dim
    ll_current = compute_bt_log_likelihood(phis_decisive, ys_decisive, omega, scale)

    samples = []
    n_accepted = 0
    for step in range(burn_in + n_samples):
        proposal = hit_and_run_simplex_step(omega, rng)
        ll_proposal = compute_bt_log_likelihood(phis_decisive, ys_decisive, proposal, scale)
        if np.log(rng.random()) < ll_proposal - ll_current:
            omega = proposal
            ll_current = ll_proposal
            if step >= burn_in:
                n_accepted += 1
        if step >= burn_in:
            samples.append(omega.copy())
    return np.array(samples), n_accepted / max(1, n_samples)


def sample_bt_posterior_hit_and_run_with_scale(
    phis_decisive: np.ndarray,
    ys_decisive: np.ndarray,
    n_samples: int = 300,
    burn_in: int = 200,
    scale_init: float = 1.0,
    scale_proposal_sd: float = 0.25,
    scale_bounds: Tuple[float, float] = (0.05, 20.0),
    scale_prior_mean: float = 0.0,    # log-normal in log-space: log(s) ~ N(0, sd^2)
    scale_prior_sd: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Jointly sample (omega, s). Mirrors sample_posterior_hit_and_run_unknown_noise:
    hit-and-run for omega, log-space random walk MH for s with log-normal prior."""
    if rng is None:
        rng = np.random.default_rng()
    dim = DIM
    omega = np.ones(dim) / dim
    s = float(scale_init)

    def loglik(om, sc):
        return compute_bt_log_likelihood(phis_decisive, ys_decisive, om, sc)

    def log_prior_s(sc):
        if sc <= 0:
            return -np.inf
        z = (np.log(sc) - scale_prior_mean) / scale_prior_sd
        return -0.5 * z * z - np.log(sc)   # log-normal density (constants dropped)

    ll_current = loglik(omega, s)
    omega_samples, s_samples = [], []
    n_acc_omega = n_acc_s = 0

    for step in range(burn_in + n_samples):
        # omega update
        proposal = hit_and_run_simplex_step(omega, rng)
        ll_prop = loglik(proposal, s)
        if np.log(rng.random()) < ll_prop - ll_current:
            omega = proposal
            ll_current = ll_prop
            if step >= burn_in:
                n_acc_omega += 1
        # scale update (random walk in log-space)
        s_prop = float(np.exp(np.log(s) + rng.normal(0.0, scale_proposal_sd)))
        if scale_bounds[0] <= s_prop <= scale_bounds[1]:
            ll_s_prop = loglik(omega, s_prop)
            log_acc = (ll_s_prop - ll_current
                       + log_prior_s(s_prop) - log_prior_s(s))
            if np.log(rng.random()) < log_acc:
                s = s_prop
                ll_current = ll_s_prop
                if step >= burn_in:
                    n_acc_s += 1
        if step >= burn_in:
            omega_samples.append(omega.copy())
            s_samples.append(s)

    info = {'omega_accept_rate': n_acc_omega / max(1, n_samples),
            'scale_accept_rate': n_acc_s / max(1, n_samples)}
    return np.array(omega_samples), np.array(s_samples), info


# ----------------------------------------------------------------------------
# 3. Trial runner — mirrors run_multiframe_trial structure exactly
# ----------------------------------------------------------------------------
def run_bt_trial_hitandrun(
    oracle_weights: np.ndarray,
    noise_type: str,
    scale_delta: float,
    scale_r: float,
    tau: float,
    tau_prime: float,
    lambda_x: float,
    n_attempts: int,
    forced_choice: Optional[str] = None,    # None | 'random' | 'left' | 'lex'
    lex_ranking: Optional[List[int]] = None,
    n_candidates: int = 50,
    n_posterior_samples: int = 200,
    burn_in: int = 200,
    holdout: Optional[List[Tuple[PairwiseQuery, str]]] = None,
    learn_scale: bool = True,
    fixed_scale: float = 1.0,    # used when learn_scale=False (e.g., 1/scale_delta to match a known logistic scale)
    V: np.ndarray = None,
    rng: Optional[np.random.Generator] = None,
    candidate_rng: Optional[np.random.Generator] = None,
    noise_fn: Optional[Callable] = None,
) -> Dict:
    """
    BT trial with hit-and-run MCMC posterior. Same surface area as
    run_bt_trial_laplace_bald, same loop structure as run_multiframe_trial.

    The acquisition is BALD-bernoulli over MCMC samples — identical estimator
    shape to multi-frame's BALD-categorical, just over 2 outcomes instead of 4.

    Note: BT does NOT use tau / tau_prime internally — they only drive the
    oracle DGP (passed through to predict_response_noisy).
    """
    if rng is None:
        rng = np.random.default_rng()
    dim = len(oracle_weights)

    if noise_fn is None:
        noise_fn = create_noise_fn(noise_type, scale_delta, scale_r, rng)

    # storage
    transcript = []
    phis_decisive: List[np.ndarray] = []
    ys_decisive:   List[float]      = []
    cos_sims, l1s, holdout_lls = [], [], []
    response_sequence = []
    linf_diameters, l1_diameters = [], []
    responses, forced_responses = Counter(), Counter()
    n_forced = 0

    # initial posterior = uniform Dirichlet samples (matches multi-frame init)
    omega_samples = rng.dirichlet(np.ones(dim), size=n_posterior_samples)
    scale = 1.0 if learn_scale else fixed_scale
    scale_samples = np.full(n_posterior_samples, scale)

    for t in range(n_attempts):
        # --- 1. propose candidates (shared rng with multi-frame) ---
        candidates = generate_candidate_queries(n_candidates, candidate_rng or rng)

        # --- 2. score each by BALD-bernoulli over current MCMC samples ---
        # If learn_scale, average over (omega, s) joint samples by passing
        # bald_bernoulli_from_samples a per-sample scale (cheap loop OK at this size).
        best_q, best_score = None, -np.inf
        for q in candidates:
            phi_vec = V @ phi(q) if V is not None else phi(q)
            if learn_scale:
                # Vectorize: predictive p(left|s_i, omega_i, phi)
                logits = scale_samples * (omega_samples @ phi_vec)
                probs  = sigmoid(logits)
                mean_p = probs.mean()
                H_mean = bernoulli_entropy(np.array([mean_p]))[0]
                mean_H = bernoulli_entropy(probs).mean()
                score  = H_mean - mean_H
            else:
                score = bald_bernoulli_from_samples(phi_vec, omega_samples, scale)
            if score > best_score:
                best_score, best_q = score, q
        if best_q is None:
            best_q = candidates[0]

        # --- 3. query oracle (same DGP path as multi-frame) ---
        raw = predict_response_noisy(
            best_q, oracle_weights, noise_fn, tau, lambda_x, tau_prime, V=V,
        )
        transcript.append((best_q, raw))
        responses[raw] += 1
        response_sequence.append(raw)

        # --- 4. resolve forced-choice exactly like the Laplace runner ---
        if raw in ('left', 'right'):
            response = raw
        elif forced_choice is None:
            response = raw          # tracked but skipped for inference
        elif forced_choice == 'random':
            response = 'left' if rng.random() < 0.5 else 'right'
            n_forced += 1
        elif forced_choice == 'left':
            response = 'left'
            n_forced += 1
        elif forced_choice in ('lex', 'lexicographic'):
            response = lex_choice(best_q, ranking=lex_ranking, rng=rng)
            n_forced += 1
        else:
            raise ValueError(f"Unknown forced_choice mode: {forced_choice}")
        forced_responses[response] += 1

        # --- 5. fold decisive responses in, refit posterior via hit-and-run ---
        new_data = response in ('left', 'right')
        if new_data:
            phi_vec = V @ phi(best_q) if V is not None else phi(best_q)
            phis_decisive.append(phi_vec)
            ys_decisive.append(1.0 if response == 'left' else 0.0)

            phis_arr = np.array(phis_decisive)
            ys_arr   = np.array(ys_decisive)

            if learn_scale:
                omega_samples, scale_samples, _ = sample_bt_posterior_hit_and_run_with_scale(
                    phis_arr, ys_arr,
                    n_samples=n_posterior_samples, burn_in=burn_in, rng=rng,
                )
                scale = float(scale_samples.mean())
            else:
                omega_samples, _ = sample_bt_posterior_hit_and_run(
                    phis_arr, ys_arr,
                    n_samples=n_posterior_samples, burn_in=burn_in,
                    scale=fixed_scale, rng=rng,
                )
                scale_samples = np.full(n_posterior_samples, fixed_scale)
                scale = fixed_scale
        # else: posterior unchanged (no new info)

        # --- 6. metrics (sample-mean point estimate, like multi-frame) ---
        omega_mean = omega_samples.mean(axis=0)
        cos_sims.append(cosine_similarity(omega_mean, oracle_weights))
        l1s.append(l1_error(omega_mean, oracle_weights))

        from scipy.spatial.distance import pdist
        linf_diameters.append(float(np.max(np.ptp(omega_samples, axis=0))))
        l1_diameters.append(float(np.max(pdist(omega_samples, metric='cityblock'))))

        if holdout is not None:
            ll = bt_log_loss(omega_mean, holdout, scale=scale, V=V)
            holdout_lls.append(ll)

    out = {
        'cos_sims': cos_sims,
        'l1s': l1s,
        'responses': responses,
        'forced_responses': forced_responses,
        'n_decisive': len(phis_decisive),
        'n_forced': n_forced,
        'response_sequence': response_sequence,
        'linf_diameters': linf_diameters,
        'l1_diameters': l1_diameters,
        'final_estimate': omega_samples.mean(axis=0),
        'scale': scale,
    }
    if holdout is not None:
        out['holdout_ll'] = holdout_lls
    return out


# ----------------------------------------------------------------------------
# 4. Wire it into the dispatcher (run_cell_experiment, cell 8)
# ----------------------------------------------------------------------------
# Add a branch like:
#
#   elif method == 'bt_hitandrun':
#       result = run_bt_trial_hitandrun(
#           oracle_weights=oracle_w, noise_type=noise_type,
#           scale_delta=scale_delta, scale_r=scale_r,
#           tau=tau, tau_prime=tau_prime, lambda_x=lambda_x,
#           n_attempts=T, n_candidates=n_candidates,
#           n_posterior_samples=n_posterior_samples,
#           holdout=holdout, learn_scale=learn_scale, V=V,
#           rng=trial_rng,
#           candidate_rng=candidate_rng,    # shared with multiframe!
#           noise_fn=noise_fn,              # shared with multiframe!
#       )
#
# And add 'bt_hitandrun' (and any forced-choice variants you want, e.g.
# 'bt_hitandrun_random', 'bt_hitandrun_lex') to the methods list in
# run_big_sweep.py / run_test_sweep.py.
#
# At tau=tau'=0 with logistic noise, this should overlap multiframe up to
# MCMC noise — the likelihoods coincide, the sampler is identical, and the
# BALD estimator is the same shape (just 2 outcomes vs 4 that collapse to 2).
