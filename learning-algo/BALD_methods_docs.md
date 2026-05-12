# BALD Methods — Algorithm Documentation

Eight active-learning methods compared in `BALD_bt_vs_multiframe_experiment.ipynb`.
Each method learns a weight vector $\omega \in \Delta^{d-1}$ over $d$ priorities
from sequential pairwise queries to an oracle.

The methods split into two groups by **what they assume about the oracle's
noise model**:

- **Known noise** (methods 1–6): the learner knows the noise family and shape.
- **Noise not known** (methods 7–8): the learner must jointly infer the
  noise distribution along with $\omega$.

---

## 0. Common scaffolding

Everything below is shared across all 8 methods.

### 0.1 Data-generating process (oracle)

A pairwise query $q = (x_L, x_R)$ produces a per-frame feature gap
$g = \lambda \cdot \phi(q)$ with $\phi(q) = x_L - x_R$. The aggregate signed
score is $\Delta = \omega \cdot g$, the intensity is $r = \sum_j \omega_j |g_j|$,
and the latent score with noise is $\tilde\Delta = \Delta + \varepsilon$ where
$\varepsilon$ is drawn from a configurable noise family
(`logistic` / `normal` / `gumbel` / `gennorm`).

The outcome rule:

| condition | label |
|---|---|
| $r < \tau$ | `indifferent` |
| $r \ge \tau$ and $|\tilde\Delta| < \tau' \cdot r$ | `incomparable` |
| $\tilde\Delta \ge \tau' \cdot r$ | `left` |
| $\tilde\Delta \le -\tau' \cdot r$ | `right` |

so each query yields one of four possible responses.

### 0.2 Oracle, holdout, candidate pool

For every $(\tau, \tau')$ cell, the trial fixes:
- a sparse Dirichlet($\alpha = 0.2$) oracle $\omega^\star$ (seeded so
  the same oracle is used across all 8 methods);
- a holdout set of labelled queries (for held-out log-loss);
- a candidate pool from which the acquisition picks the next query;
- the per-trial RNG that draws the noise $\varepsilon$.

### 0.3 Acquisition (all methods)

Every method scores `n_candidates` random query pairs with the BALD criterion
$$
\text{BALD}(q) \;=\; H\!\bigl[\mathbb{E}_\omega\, p(y \mid q, \omega)\bigr]
                  \;-\; \mathbb{E}_\omega\,H\!\bigl[p(y \mid q, \omega)\bigr],
$$
i.e. mutual information between the response $y$ and the weight vector
$\omega$. The query with the largest BALD score is selected, queried, the
response is appended to the transcript, and the posterior over $\omega$ is
updated.

The methods differ in **two places**: what $y$'s response space looks like
(2- / 3- / 4-outcome), and how the posterior over $\omega$ is represented.

### 0.4 Posterior summary and metrics

The posterior is a set of samples (or a Gaussian, for the Laplace methods);
the point estimate $\hat\omega$ is the sample mean. Metrics per iteration:
cosine similarity to $\omega^\star$, $\ell_1$ error, holdout log-loss,
posterior $\ell_\infty$ / $\ell_1$ diameter.

### 0.5 Initialisation

All methods start from a fresh Dirichlet$(\mathbf 1)$ posterior (uniform on
the simplex) before the first query.

---

## 1. Known-noise methods (the noise family + shape is given)

The learner uses the same noise CDF the oracle used.

### 1.1 `multiframe` — Utilize-Indecision (4-outcome)

| field | value |
|---|---|
| Response space | 4-outcome: `left` / `right` / `indifferent` / `incomparable` |
| Likelihood | Closed-form 4-outcome probabilities via the noise CDF (`compute_response_probs_mc`). Uses the *true* $\tau, \tau'$ from the cell. |
| Posterior over $\omega$ | Hit-and-run MCMC on the simplex with Metropolis–Hastings on the full transcript log-likelihood. |
| Acquisition | 4-outcome BALD. |
| Indecision | **Retained as informative labels** — both `indifferent` (rule fires on intensity) and `incomparable` (rule fires on noise) enter the likelihood, telling the learner about $\omega$. |

This is the *ceiling*: the learner uses every bit of information in the
response.

### 1.2 `multiframe_3outcome` — Utilize-Indecision (3-outcome)

| field | value |
|---|---|
| Response space | 3-outcome: `left` / `right` / `indecisive` (collapses `indifferent` + `incomparable`) |
| Likelihood | Same noise CDF as `multiframe`, but the indecisive bucket is $p_\text{indif} + p_\text{incomp}$. |
| Posterior | Hit-and-run on the simplex with 3-outcome log-likelihood. |
| Acquisition | 3-outcome BALD. |
| Indecision | **Retained but coarsened** — the learner knows "the rule was indecisive" but not why. |

Tests how much information lives in distinguishing the two indecisive
sub-cases.

### 1.3 `bt_laplace_bald` — BT (Skip)

| field | value |
|---|---|
| Response space | 2-outcome (forced binary) |
| Likelihood | Bradley–Terry: $p(\text{left}) = \sigma(s \cdot \omega \cdot \phi(q))$ with a learned scale $s > 0$. |
| Posterior | SLSQP MAP on the simplex with an L2 prior, plus Laplace covariance $H^{-1}$ for samples. Samples are drawn from $\mathcal{N}(\hat\omega, H^{-1})$ then projected onto the simplex. |
| Acquisition | Bernoulli BALD. |
| Indecision | **Dropped.** Non-decisive responses (`indifferent` / `incomparable`) are discarded — they don't enter the BT likelihood at all. The learner sees fewer effective queries than the budget. |

The standard baseline.

### 1.4 `bt_laplace_bald_k` — BT (K-Decisive)

Identical to (1.3) but **the trial loop keeps querying until $k$ decisive
responses are collected**, instead of running for a fixed number of attempts.

| field | difference |
|---|---|
| Effective budget | Larger (cheats by running until enough decisive labels accumulate). Metrics are indexed by *decisive count*, not attempt count. |

This isolates "how informative is a decisive response" from "how many of them
the algorithm gets" — it's a fairness-of-budget probe.

### 1.5 `bt_laplace_bald_random` — BT (Random Forced Choice)

Identical to (1.3) but **non-decisive raw responses are coin-flipped to
`left`/`right`** before entering the BT likelihood.

| field | difference |
|---|---|
| Indecision | Forced to a random binary label. The likelihood treats every query as decisive. |

Pure information-loss baseline — what does BT look like if you flip a coin on
indecisive cases?

### 1.6 `bt_laplace_bald_lex` — BT (Lexicographic Forced Choice)

Identical to (1.3) but **non-decisive responses are forced by lexicographic
tie-breaking** over a fixed feature ranking. Compare $x_L, x_R$ along the
highest-priority feature first; if tied, drop to the next; if all ranked
features tie, fall back to a coin flip.

| field | difference |
|---|---|
| Indecision | Forced to a deterministic binary label by tie-breaking on a feature ranking — a structured-prior alternative to random FC. |

---

## 2. Noise-not-known methods (the noise family must also be inferred)

These methods learn a Gaussian-mixture approximation of the noise CDF
jointly with $\omega$. The learner is **not given the family** (logistic /
gennorm / etc.) the oracle used.

The mixture has $K = 3$ components, $\varepsilon \sim \sum_{k=1}^K w_k\,
\mathcal N(\mu_k, \sigma_k^2)$, with closed-form mixture CDF for the response
probabilities. Inference uses **Metropolis-within-Gibbs** to sample the joint
posterior $(\omega, w, \mu, \sigma)$:

| block | step |
|---|---|
| $\omega$ | hit-and-run on the $d$-simplex |
| $w$ | hit-and-run on the $K$-simplex (mixture weights) |
| $\mu_k$, each $k$ | random-walk Metropolis (Gaussian prior on $\mu_k$) |
| $\log \sigma_k$, each $k$ | random-walk Metropolis (log-normal prior on $\sigma_k$) |

This is heavier per iteration than the known-noise methods.

### 2.1 `multiframe_unknown_family` — Utilize-Indecision with MoG noise

| field | value |
|---|---|
| Response space | 4-outcome (same as `multiframe`) |
| Noise model | $K = 3$ Gaussian mixture, learned. |
| Likelihood | 4-outcome MF likelihood with MoG CDF replacing the assumed noise CDF. |
| Posterior | Joint M-within-Gibbs over $(\omega, w, \mu, \sigma)$. |
| Acquisition | 4-outcome BALD marginalised over the joint posterior. |
| Indecision | **Retained** (4-outcome). |

This is the noise-agnostic analogue of `multiframe`. Pays compute cost in
exchange for not knowing the noise family.

### 2.2 `bt_mog` — BT with MoG noise

| field | value |
|---|---|
| Response space | 2-outcome (BT, forced binary; **no $\tau, \tau'$**) |
| Noise model | $K = 3$ Gaussian mixture, learned. |
| Likelihood | $p(\text{left}) = 1 - F_{\text{MoG}}(-\Delta)$. |
| Posterior | Same joint M-within-Gibbs over $(\omega, w, \mu, \sigma)$ as 2.1. |
| Acquisition | Bernoulli BALD marginalised over the joint posterior. |
| Indecision | **Dropped.** Only decisive items contribute. |

The BT analogue of `multiframe_unknown_family` — same noise treatment, but
with the binary-response, indecision-discarding BT formulation.

---

## 3. Comparison tables

### 3.1 What each method retains from the response

| method | response space | indecision handling | noise model |
|---|---|---|---|
| `multiframe`              | 4-outcome | retained, full | known |
| `multiframe_3outcome`     | 3-outcome | retained, coarsened | known |
| `multiframe_unknown_family` | 4-outcome | retained, full | learned (MoG) |
| `bt_laplace_bald`         | 2-outcome | dropped | known |
| `bt_laplace_bald_k`       | 2-outcome | dropped, but budget extended | known |
| `bt_laplace_bald_random`  | 2-outcome | forced (random) | known |
| `bt_laplace_bald_lex`     | 2-outcome | forced (lexicographic) | known |
| `bt_mog`                  | 2-outcome | dropped | learned (MoG) |

### 3.2 Inference mechanism

| method | posterior representation | inference |
|---|---|---|
| `multiframe`                | sample set (MCMC) | hit-and-run on simplex |
| `multiframe_3outcome`       | sample set | hit-and-run on simplex |
| `multiframe_unknown_family` | sample set, joint $(\omega, w, \mu, \sigma)$ | M-within-Gibbs |
| `bt_laplace_bald`*          | Gaussian | MAP (SLSQP) + Laplace covariance |
| `bt_laplace_bald_k`*        | Gaussian | MAP + Laplace |
| `bt_laplace_bald_random`*   | Gaussian | MAP + Laplace |
| `bt_laplace_bald_lex`*      | Gaussian | MAP + Laplace |
| `bt_mog`                    | sample set, joint $(\omega, w, \mu, \sigma)$ | M-within-Gibbs |

\* the BT-Laplace methods learn a scale parameter $s$ jointly with $\omega$.

### 3.3 Per-iteration cost (relative)

`multiframe_unknown_family` and `bt_mog` are the heaviest (joint MCMC over
5 parameter blocks). `multiframe` and `multiframe_3outcome` are mid-weight
(hit-and-run on the simplex). The four BT-Laplace methods are lightest
(MAP optimisation + a small projection step) but most lossy on information.

---

## 4. What the comparison is designed to isolate

Within each group:

| comparison | isolates |
|---|---|
| `multiframe` vs `multiframe_3outcome` | value of distinguishing `indifferent` vs `incomparable` |
| `multiframe` vs `multiframe_unknown_family` | cost of not knowing the noise family |
| `bt_laplace_bald` vs `bt_laplace_bald_k` | role of query-budget definition |
| `bt_laplace_bald` vs `bt_laplace_bald_random` | value of *retaining* vs *coin-flipping* indecision |
| `bt_laplace_bald_random` vs `bt_laplace_bald_lex` | random vs structured forced-choice |
| `bt_laplace_bald` vs `bt_mog` | BT's own noise-agnostic version |
| `multiframe` vs `bt_laplace_bald` | the headline: Utilize-Indecision vs BT |

The two-group structure (known noise / noise not known) makes both
"how much does Utilize-Indecision help when noise is known" and "does the
gap survive when noise must be learned" answerable from the same sweep.
