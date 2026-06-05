# final_model — Section 3 experiments (indecision in pairwise-comparison learning)

A clean, focused reorganization of `../learning-algo/BALD_bt_vs_multiframe_experiment.ipynb`.
It tests one thing: the role of indecision in pairwise-comparison learning, in three blocks.

## Files

| File | Purpose |
|---|---|
| `indecision_core.py` | The model: threshold DGP, response likelihoods, hit-and-run MCMC, BALD selection, forcing rules, metrics. No experiment logic. |
| `experiments.py` | Unified trial runners (broad 4-/3-way and binary BT) + multi-trial averaging harness. Every learner shares the MCMC sampler; only the outcome alphabet differs. |
| `final_experiments.ipynb` | The deliverable. Configures and runs Blocks 0–2 and produces the three figures. |
| `_build_notebook.py` | Source that generates `final_experiments.ipynb`. **Edit the notebook here**, then re-run `py _build_notebook.py`, or edit the `.ipynb` directly (but then this builder is stale). |
| `fig_block{0,1,2}_*.{pdf,png}`, `fig_severe_*.{pdf,png}` | Saved figures (also embedded in the notebook). |

## Model (paper ↔ code)

`delta = x_left - x_right`; `g = <omega*, delta>` (directional evidence);
`r = <omega*, |delta|>` (total evidence). Response is **indifferent** if `r < tau_r`,
**conflict** if `|g + eps| <= tau_kappa * r`, else **left/right**. At `tau_r = tau_kappa = 0`
the model is ordinary logistic Bradley–Terry with slope `1/noise_scale`.

## The three blocks

0. **Reduction** — at `tau_r=tau_kappa=0`, broad and binary learning coincide (exactly under a
   shared inference seed; within standard error otherwise).
1. **Biased forcing fails** — with `tau>0`, a binary learner fed forced labels recovers a
   systematically distorted `omega` (weight migrates onto whatever feature the forcing rule keys
   on) and chooses worse downstream rules — *unless* the forcing is benign (50/50 or BT-consistent).
   Forcing rules fall into three types: **benign** (50/50, BT-consistent); **feature-keyed biased**
   (lexicographic, single-feature, self-similarity, and `gut-weights` = defer to a different
   `omega_bias`, which generalizes the first two); and **structural** (`compromise`/extremeness
   aversion). Finding: feature-keyed rules are damaging (their signal aligns with feature
   differences), but the structural rule is near-harmless — its signal is ~orthogonal to feature
   differences, so it barely corrupts linear weight recovery. Not all heuristics are equally bad.
2. **Benign forcing wastes information** — even under unbiased forcing, observing indecision
   (3-way or 4-way) reaches a target weight error with far fewer queries.
3. **Severe regime** — Blocks 1–2 are the *benign end* (distinct options, so most responses are
   decisive). The *severe end* keeps the oracles, `tau`, and rules **identical** and changes only one
   thing: **similar options** (`query_sigma`, so `x_right = x_left + N(0,sigma)`). Indecision then
   dominates (~25% decisive), and biased forcing **inflates the weight of whatever feature its
   tiebreaker keys on** (~2× the true weight) → ~3× the downstream regret of benign forcing, while
   broad learning stays near-perfect. Nothing is engineered against the respondent; the tiebreaker is
   orthogonal to true preference, not opposed to it. (`make_adversarial_oracles` exists as an optional
   sharper variant — bias feature's true weight set to zero — but is *not* used in the main figures,
   since the effect is real without it.)

Downstream regret is reported as a **percent of the worst-case** (always-pick-truly-worst) ceiling,
computed per oracle, so "how bad is bad" is interpretable (the ceiling here is ~0.45, not 1.0).

### Robustness (appendix)

The notebook's appendix sweeps the oracle concentration (Dirichlet `alpha`), the query distribution
(distinct vs similar), and the **feature model** — cross-feature correlation (Gaussian copula via
`sample_queries(..., cov=...)`) and heterogeneous per-feature scale (`scale=...`). Findings:

* **Reduction** and **broad ≫ binary** are invariant to all of these (broad recovers `omega` at every
  setting; the τ=0 identity holds exactly even under correlated/scaled features).
* **benign < biased** holds at every `alpha`, so it is *not* an artifact of the sparse-oracle choice.
  **But the *magnitude* of the bias is largely an artifact of the i.i.d.-feature assumption.** The
  appendix builds a feature model from the **real kidney options** (`kidneystudy/kidney_features_raw.csv`
  — empirical per-attribute scales + 5×5 correlation) and, crucially, sweeps *which* attribute the
  heuristic keys on (section C — the **average case**, not the low-spread best case the main figures
  use). Under independent features the bias is target-agnostic and large (single-feature ~17% of
  worst-case ≈ 3× benign for *any* attribute); at the **real kidney geometry it averages ~6% ≈ benign**,
  with only the single highest-spread attribute (workhours) reaching ~2× benign. So the forced-choice
  catastrophe is the *fragile* result; **broad ≫ binary is the robust one** — lead with that in the paper.
* **But mild decision regret ≠ no harm (section D, `fig_correlation_masking`).** Correlation does *not*
  stop the biased rule from corrupting `omega_hat` — the learned weight on the keyed feature stays
  inflated at *every* `rho`. It only makes that distortion irrelevant to decisions *on the elicitation
  distribution* (where, at high `rho`, the weights are barely identified). Evaluate the same `omega_hat`
  on an **off-axis** deployment (independent items) and the full ~17% regret returns. So the bias is a
  **latent deployment-shift liability**: safe-looking only if you deploy to a population like the
  elicitation one; any subgroup/cohort/policy that decorrelates the attributes re-exposes it. This is
  why parameter-level metrics (weight error, per-feature distortion) and off-axis / worst-case
  deployment regret are worth reporting alongside matched-deployment regret.

## Running

```
py -m nbconvert --to notebook --execute --inplace final_experiments.ipynb
```

The notebook runs end-to-end at **pilot** scale (~1 min). For paper-quality runs, increase
`N_ORACLES`, the `T*` budgets, and `N_SAMPLES` in the setup cell — the seed plumbing and method
definitions are unchanged. Set `N_JOBS = 1` in the setup cell to disable joblib parallelism when
debugging.

Requires `numpy`, `scipy`, `matplotlib`, `joblib` (and `nbconvert`/`ipykernel` to execute).
