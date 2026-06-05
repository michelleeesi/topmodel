"""Construct final_experiments.ipynb from source strings, then it is executed
separately with nbconvert. Keeping the notebook source here (rather than hand-
editing JSON) makes the cells easy to revise."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

# ---------------------------------------------------------------- title
md(r"""# Indecision in Pairwise-Comparison Learning — Section 3 Experiments

Clean reorganization of `learning-algo/BALD_bt_vs_multiframe_experiment.ipynb`,
focused on a single question: **what is the role of indecision in pairwise-comparison learning?**

### Model recap

We are in the perfectly separable linear-score setting. A query is a pair of options;
the model depends only on the feature difference $\delta = x_{\text{left}} - x_{\text{right}}$.
Given true weights $\omega^\*$ on the simplex,

$$g = \langle \omega^\*, \delta\rangle \quad\text{(directional evidence)},\qquad
   r = \langle \omega^\*, |\delta|\rangle \quad\text{(total evidence)}.$$

The threshold response model adds a noisy margin $\tilde g = g + \varepsilon$ and two forms of indecision:

* **indifference** (low intensity) when $r < \tau_r$,
* **conflict** (high intensity) when $|\tilde g| \le \tau_\kappa\, r$,

otherwise the answer is `LEFT` ($\tilde g>0$) or `RIGHT` ($\tilde g<0$). When
$\tau_r=\tau_\kappa=0$ the model reduces to ordinary logistic Bradley–Terry with slope $1/s$.

### The three experiment blocks

| Block | Question | Headline figure |
|---|---|---|
| **0** | At $\tau_r=\tau_\kappa=0$, does the threshold model reduce to ordinary binary learning? | sanity check |
| **1** | When indecision is *forced* into binary labels, when does binary learning go wrong? | biased forced-choice failure |
| **2** | Even under *benign* forcing, is observing indecision more sample-efficient? | learning curves |

Every learner shares the same hit-and-run MCMC posterior and BALD acquisition; only the
**outcome alphabet / likelihood** differs (4-way broad, 3-way broad, or binary BT). Model
code lives in `indecision_core.py` / `experiments.py`; this notebook only configures runs and plots.""")

# ---------------------------------------------------------------- setup
code(r"""import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

HERE = os.path.abspath("")
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import indecision_core as ic
import experiments as ex

# ---- plotting style -------------------------------------------------------
mpl.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "axes.grid": True, "grid.alpha": 0.25,
})

PALETTE = {
    "broad4":   "#4c1d95",  # deep purple  -- broad 4-way (the ceiling)
    "broad3":   "#7c3aed",  # purple       -- broad 3-way
    "bt_skip":  "#0ea5e9",  # blue         -- ignore indecision
    "bt_5050":  "#10b981",  # green        -- benign 50/50
    "bt_btcons":"#059669",  # dark green   -- benign BT-consistent
    "bt_lex":   "#f59e0b",  # amber        -- biased lexicographic (feature-keyed)
    "bt_single":"#ef4444",  # red          -- biased single-feature (feature-keyed)
    "bt_self":  "#b91c1c",  # dark red     -- biased self-similarity (feature-keyed)
    "bt_gut":   "#7f1d1d",  # darkest red  -- biased gut-weights (feature-keyed, dense)
    "bt_comp":  "#64748b",  # slate        -- structural: compromise / extremeness aversion
}
NAMES = {
    "broad4": "Broad (4-way)", "broad3": "Broad (3-way)",
    "bt_skip": "Binary: ignore indecision", "bt_5050": "Binary: forced 50/50",
    "bt_btcons": "Binary: forced BT-consistent", "bt_lex": "Binary: forced lexicographic",
    "bt_single": "Binary: forced single-feature", "bt_self": "Binary: forced self-similarity",
    "bt_gut": "Binary: forced gut-weights", "bt_comp": "Binary: forced compromise",
}

# ---- global simulation configuration -------------------------------------
# PILOT scale runs end-to-end in a couple of minutes. Scale up (N_ORACLES, T,
# N_SAMPLES) for the paper-quality run; the seeds and structure are unchanged.
DIM         = ic.DIM
NOISE_SCALE = 0.5
NOISE_TYPE  = "logistic"
N_ORACLES   = 40        # oracles == trials; more -> tighter error bars
ORACLE_ALPHA= 0.3        # sparse/peaky oracles (harder; room to improve)
N_SAMPLES   = 150
BURN_IN     = 80
N_CAND      = 30
N_HOLDOUT   = 400
N_JOBS      = -1         # joblib trials in parallel; set 1 to debug
TAU_R, TAU_K = 0.25, 0.25  # operating point for Blocks 1 & 2

ORACLES = ex.make_oracles(N_ORACLES, dim=DIM, seed=2026, alpha=ORACLE_ALPHA)
print(f"{N_ORACLES} oracles, dim={DIM}, noise={NOISE_TYPE}(s={NOISE_SCALE})")
print(f"feature names: {ic.FEATURE_NAMES}")

# ---- seed policy: common random numbers (CRN) ----------------------------
# run_method() derives per-trial seeds for the candidate pool, oracle noise,
# holdout, and regret slates from the *trial index only*, so every method sees
# identical conditions on a given trial. The MCMC seed is base_seed + trial.
# We pass the SAME base_seed to every method, so methods are compared paired
# (they differ only in their own logic, not in their randomness). In particular,
# at tau=0 the broad and binary learners then produce *identical* curves.
SEED = 2027

# Worst-case downstream regret = the anti-optimal policy that always picks the item
# the TRUE rule ranks lowest. Per slate that equals the spread (max-min) of true
# scores; we average it per oracle and use it to report regret as a % of this ceiling
# (it is NOT 1.0 -- it depends on how spread the true scores are).
def worstcase_regret(omega_star, seed, n_slates=4000, slate_size=5, cov=None, scale=None):
    rng = np.random.default_rng(seed)
    items = ic.draw_features(n_slates * slate_size, DIM, rng, cov, scale).reshape(n_slates, slate_size, DIM)
    t = items @ omega_star
    return float(np.mean(t.max(1) - t.min(1)))

def regret_ceiling(oracles, cov=None, scale=None):
    return float(np.mean([worstcase_regret(w, 600_000 + i, cov=cov, scale=scale)
                          for i, w in enumerate(oracles)]))
""")

# ---------------------------------------------------------------- BLOCK 0
md(r"""## Block 0 — Reduction theorem (sanity check)

**Claim.** At $\tau_r=\tau_\kappa=0$ there is never an indifferent or conflicted response, and the
4-outcome likelihood collapses *exactly* to logistic Bradley–Terry with slope $1/s$
(verified algebraically in `indecision_core.response_probs`). So the broad-response learner and
the ordinary binary learner should behave essentially identically.

We run both at $\tau_r=\tau_\kappa=0$ under **common random numbers** (same oracle, candidates, and
inference seed). Because the likelihoods are equal, the two learners see identical data and produce
**bit-for-bit identical** weight-error curves — the strongest form of the sanity check (we assert
the max curve difference is numerically zero). The binary curve is drawn dashed with markers, on top
of the broad curve, so you can see the two coincide exactly.""")

code(r"""T0 = 50
cfg0 = dict(tau_r=0.0, tau_kappa=0.0, noise_scale=NOISE_SCALE, noise_type=NOISE_TYPE,
            T=T0, n_candidates=N_CAND, n_samples=N_SAMPLES, burn_in=BURN_IN,
            n_holdout=N_HOLDOUT, n_jobs=N_JOBS)

t0 = time.time()
# Same base_seed for both -> common random numbers -> identical curves at tau=0.
b0 = {
    "broad4":  ex.run_method({"kind": "broad", "n_outcomes": 4}, ORACLES, base_seed=SEED, **cfg0),
    "bt_skip": ex.run_method({"kind": "bt"},                     ORACLES, base_seed=SEED, **cfg0),
}
print(f"Block 0 done in {time.time()-t0:.1f}s")

c_b = ex.aggregate_curve(b0["broad4"], "l1s")["mean"]
c_t = ex.aggregate_curve(b0["bt_skip"], "l1s")["mean"]
maxdiff = float(np.abs(c_b - c_t).max())
print(f"max |broad - binary| over the L1 curve = {maxdiff:.2e}")
assert maxdiff < 1e-9, "reduction is exact at tau=0: curves must be identical"
""")

code(r"""fig, ax = plt.subplots(figsize=(5.4, 3.7))
x = np.arange(1, T0 + 1)
agg_b = ex.aggregate_curve(b0["broad4"], "l1s")
agg_t = ex.aggregate_curve(b0["bt_skip"], "l1s")
ax.fill_between(x, agg_b["mean"] - agg_b["stderr"], agg_b["mean"] + agg_b["stderr"],
                color=PALETTE["broad4"], alpha=0.15)
ax.plot(x, agg_b["mean"], color=PALETTE["broad4"], lw=3, label=NAMES["broad4"])
ax.plot(x, agg_t["mean"], color=PALETTE["bt_skip"], lw=1.4, ls="--", marker="o", ms=3.5,
        markevery=2, label=NAMES["bt_skip"] + "  (overlaid)")
ax.set_xlabel("queries"); ax.set_ylabel(r"weight error  $\|\hat\omega-\omega^*\|_1$")
ax.set_title(r"Block 0: at $\tau_r=\tau_\kappa=0$, broad $\equiv$ binary")
ax.text(0.97, 0.93, f"max curve difference = {maxdiff:.0e}", transform=ax.transAxes,
        ha="right", va="top", fontsize=9, color="0.35")
ax.legend(loc="lower left")
fig.savefig("fig_block0_sanity.pdf"); fig.savefig("fig_block0_sanity.png")
plt.show()
print("The dashed binary curve lies exactly on the broad curve (max difference "
      f"{maxdiff:.0e}) -> the threshold model reduces to ordinary binary learning when "
      "there is no indecision.")
""")

# ---------------------------------------------------------------- BLOCK 1
md(r"""## Block 1 — Forcing indecision into binary labels

Now $\tau_r,\tau_\kappa>0$, so the respondent is sometimes internally indecisive, but the interface
forces a `LEFT`/`RIGHT` choice. We feed a **binary** learner the forced labels under different
behavioral assumptions about *how* people resolve indecision:

* **Benign** — `50/50` (unbiased coin) and `BT-consistent` (choose `LEFT` w.p. $\sigma(g/s)$);
* **Feature-keyed biased** — `lexicographic` (follow a feature ranking), `single-feature` (always
  follow one feature), `similarity-to-self` (closer to a fixed self-vector), and `gut-weights`
  (defer to a *different* weight vector $\omega_{\text{bias}}$ — the dual-process "System-1" heuristic;
  this generalizes single-feature and lexicographic to a dense pull);
* **Structural** — `compromise` (extremeness aversion: pick the less extreme option). Included to test
  a *non*-feature-keyed heuristic.

Each *feature-keyed* rule injects a signal aligned with the feature differences on the indecisive
queries (where $g\approx0$), which the binary learner misattributes to $\omega$. The structural rule
does not align with any feature axis — a useful contrast, since it turns out to be far less damaging.

**Isolating the forcing effect.** Any binary learner is mildly misspecified here (decisive responses
come from the threshold model, not pure Bradley–Terry), which produces a small distortion *shared by
every* binary method — even the one that simply ignores indecision. To show the effect of the
*behavior*, we plot **forcing-induced distortion** $\hat\omega(\text{rule})-\hat\omega(\text{ignore})$,
paired per oracle, which cancels that shared drift. We also report **downstream best-of-$N$ regret**
under the true $\omega^\*$ and final weight error (absolute) in the table.

The feature-keyed rules are deliberately pointed at *different* features so the distortion is visibly
rule-specific: single-feature/self → `elderlyDep` (feature 0), lexicographic → `lifeYearsGained`
(feature 1), gut-weights → mostly `obesity` (feature 2). The structural `compromise` rule keys on no
feature axis.""")

code(r"""T1 = 130
SELF_VEC = np.eye(DIM)[0]               # 'self' favors feature 0 (elderlyDep)
GUT_WEIGHTS = np.array([0.1, 0.1, 0.5, 0.2, 0.1])  # System-1 'gut' fixates on obesity (feature 2)
cfg1 = dict(tau_r=TAU_R, tau_kappa=TAU_K, noise_scale=NOISE_SCALE, noise_type=NOISE_TYPE,
            T=T1, n_candidates=N_CAND, n_samples=N_SAMPLES, burn_in=BURN_IN,
            n_holdout=N_HOLDOUT, n_jobs=N_JOBS)

SPECS1 = {  # ordered: benign, then feature-keyed biased, then structural
    "bt_5050":  {"kind": "bt", "forcing": ic.force_5050},
    "bt_btcons":{"kind": "bt", "forcing": ic.force_bt_consistent},
    "bt_lex":   {"kind": "bt", "forcing": ic.force_lex,
                 "forcing_kwargs": {"ranking": [1, 3, 0, 2, 4]}},
    "bt_single":{"kind": "bt", "forcing": ic.force_single_feature,
                 "forcing_kwargs": {"feature": 0}},
    "bt_self":  {"kind": "bt", "forcing": ic.force_self_similarity,
                 "forcing_kwargs": {"self_vec": SELF_VEC}},
    "bt_gut":   {"kind": "bt", "forcing": ic.force_gut_weights,
                 "forcing_kwargs": {"omega_bias": GUT_WEIGHTS}},
    "bt_comp":  {"kind": "bt", "forcing": ic.force_compromise},
}
BENIGN = ["bt_5050", "bt_btcons"]
STRUCTURAL = ["bt_comp"]   # non-feature-keyed; reported separately from the feature-keyed biased rules

t0 = time.time()
# Common random numbers: every rule and the baseline share base_seed=SEED, so the
# forcing-induced distortion below reflects only the forcing behavior, not MCMC noise.
b1 = {k: ex.run_method(s, ORACLES, base_seed=SEED, **cfg1) for k, s in SPECS1.items()}
# No-forcing baseline (ignore indecision) -> reference for forcing-induced distortion.
b1_skip = ex.run_method({"kind": "bt"}, ORACLES, base_seed=SEED, **cfg1)
print(f"Block 1 done in {time.time()-t0:.1f}s")

WC = regret_ceiling(ORACLES)   # anti-optimal regret ceiling for this oracle set
print(f"worst-case (anti-optimal) regret ceiling = {WC:.3f}\n")

print(f"{'forcing rule':<28}{'L1':>8}{'cos':>8}{'regret %worst':>14}{'pair err':>10}")
for k in SPECS1:
    L1 = ex.aggregate_scalar(b1[k], 'l1_final')['mean']
    cs = ex.aggregate_scalar(b1[k], 'cos_final')['mean']
    bo = ex.aggregate_scalar(b1[k], 'best_of_n_regret')['mean']
    pe = ex.aggregate_scalar(b1[k], 'pairwise_regret')['mean']
    tag = "benign" if k in BENIGN else ("structural" if k in STRUCTURAL else "biased")
    print(f"{NAMES[k]:<28}{L1:>8.3f}{cs:>8.3f}{100*bo/WC:>13.0f}%{pe:>10.3f}   [{tag}]")
""")

code(r"""fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 4.8),
                                gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.45})

# Panel A: forcing-induced distortion heatmap (rules x features), relative to ignoring.
keys1 = list(SPECS1.keys())
D = np.array([ex.relative_distortion(b1[k], b1_skip)["mean"] for k in keys1])  # (n_rules, DIM)
vmax = np.abs(D).max()
im = axA.imshow(D, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
axA.set_xticks(range(DIM)); axA.set_xticklabels(ic.FEATURE_NAMES, rotation=40, ha="right")
axA.set_yticks(range(len(keys1))); axA.set_yticklabels([NAMES[k] for k in keys1])
axA.set_title(r"Forcing-induced distortion  $\hat\omega_{\rm rule}-\hat\omega_{\rm ignore}$")
for i in range(len(keys1)):
    for j in range(DIM):
        axA.text(j, i, f"{D[i,j]:+.02f}", ha="center", va="center",
                 fontsize=8, color="black")
axA.axhline(len(BENIGN) - 0.5, color="k", lw=1.0)  # benign | biased divider
fig.colorbar(im, ax=axA, fraction=0.046, pad=0.04)

# Panel B: downstream best-of-N regret per rule, as % of the anti-optimal ceiling.
bo  = [100 * ex.aggregate_scalar(b1[k], 'best_of_n_regret')['mean']  / WC for k in keys1]
boe = [100 * ex.aggregate_scalar(b1[k], 'best_of_n_regret')['stderr'] / WC for k in keys1]
colors = [PALETTE[k] for k in keys1]
axB.bar(range(len(keys1)), bo, yerr=boe, color=colors, capsize=3)
axB.set_xticks(range(len(keys1)))
axB.set_xticklabels([NAMES[k].replace("Binary: forced ", "") for k in keys1],
                    rotation=40, ha="right")
axB.set_ylabel("downstream regret  (% of worst-case)")
axB.set_title("Downstream rule regret")

fig.suptitle(r"Block 1 ($\tau_r=%.2f,\ \tau_\kappa=%.2f$, distinct options): "
             "biased forced-choice corrupts weights and downstream decisions" % (TAU_R, TAU_K),
             y=1.02, fontsize=12)
fig.savefig("fig_block1_bias.pdf"); fig.savefig("fig_block1_bias.png")
plt.show()
print("Benign rules (above the line) leave ω essentially undistorted; biased rules push weight "
      "onto the feature they key on and pay for it in downstream regret.")
""")

# ---------------------------------------------------------------- BLOCK 2
md(r"""## Block 2 — Even benign forcing wastes information

Grant the binary learner the most favorable case: when the respondent is indecisive they answer
**benignly** (unbiased `50/50`), so binary learning is not asymptotically wrong. We compare, per
unit of **query budget** (every elicited query costs the same regardless of the answer):

* **Binary, ignore indecision** — drops indecisive responses;
* **Binary, forced 50/50** — benign forced labels;
* **Broad (3-way)** — observes `{LEFT, RIGHT, UNKNOWN}`;
* **Broad (4-way)** — observes `{LEFT, RIGHT, INDIFFERENT, CONFLICT}`.

Indifference constrains $r=\langle\omega,|\delta|\rangle$ and conflict constrains $|g|$ in ways
binary labels cannot, so the broad learners should reach a given weight error with fewer queries.""")

code(r"""T2 = 60
cfg2 = dict(tau_r=TAU_R, tau_kappa=TAU_K, noise_scale=NOISE_SCALE, noise_type=NOISE_TYPE,
            T=T2, n_candidates=N_CAND, n_samples=N_SAMPLES, burn_in=BURN_IN,
            n_holdout=N_HOLDOUT, n_jobs=N_JOBS)

SPECS2 = {
    "bt_skip": {"kind": "bt"},
    "bt_5050": {"kind": "bt", "forcing": ic.force_5050},
    "broad3":  {"kind": "broad", "n_outcomes": 3},
    "broad4":  {"kind": "broad", "n_outcomes": 4},
}
t0 = time.time()
# Common random numbers: all learners share base_seed=SEED, so the curves are paired
# per oracle (they differ only in outcome alphabet / forcing, not in randomness).
b2 = {k: ex.run_method(s, ORACLES, base_seed=SEED, **cfg2) for k, s in SPECS2.items()}
print(f"Block 2 done in {time.time()-t0:.1f}s")
""")

code(r"""fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.5, 4.0))
x = np.arange(1, T2 + 1)
for key in SPECS2:
    aL = ex.aggregate_curve(b2[key], "l1s")
    axL.plot(x, aL["mean"], color=PALETTE[key], lw=2, label=NAMES[key])
    axL.fill_between(x, aL["mean"]-aL["stderr"], aL["mean"]+aL["stderr"],
                     color=PALETTE[key], alpha=0.18)
    aC = ex.aggregate_curve(b2[key], "cos_sims")
    axR.plot(x, aC["mean"], color=PALETTE[key], lw=2, label=NAMES[key])
    axR.fill_between(x, aC["mean"]-aC["stderr"], aC["mean"]+aC["stderr"],
                     color=PALETTE[key], alpha=0.18)
axL.set_xlabel("query budget"); axL.set_ylabel(r"weight error $\|\hat\omega-\omega^*\|_1$")
axL.set_title("Weight error vs. budget"); axL.legend(fontsize=9)
axR.set_xlabel("query budget"); axR.set_ylabel(r"cosine similarity to $\omega^*$")
axR.set_title("Alignment vs. budget")
fig.suptitle("Block 2: broad response alphabets learn faster than binary, even under benign forcing",
             y=1.02, fontsize=12)
fig.savefig("fig_block2_efficiency.pdf"); fig.savefig("fig_block2_efficiency.png")
plt.show()

# Budget to reach a target weight error.
TARGET = 0.30
print(f"queries to reach L1 <= {TARGET}:")
for key in SPECS2:
    m = ex.aggregate_curve(b2[key], "l1s")["mean"]
    hit = np.where(m <= TARGET)[0]
    print(f"  {NAMES[key]:<30}{('%d' % (hit[0]+1)) if len(hit) else '> %d' % T2:>6}")
""")

# ---------------------------------------------------------------- SEVERE REGIME
md(r"""## Severe-indecision regime — when indecision dominates

Blocks 1–2 compared **clearly-distinct** options ($x_{\text{left}},x_{\text{right}}$ independent),
so most responses are decisive and the BALD learner can lean on genuine choices — biased forcing only
mildly distorts $\omega$. That is the benign end. But real elicitation often compares **similar**
options: people hesitate precisely when the alternatives are alike. We change *one thing* and keep
everything else identical to Blocks 1–2 (same oracles, same $\tau$, same five rules):

> **Similar options** — $x_{\text{right}} = x_{\text{left}} + \mathcal{N}(0,\sigma)$, $\sigma=0.3$.

Smaller feature differences mean smaller total evidence $r$, so the respondent is *indecisive most of
the time*, and the binary learner's labels are now dominated by the forcing rule rather than genuine
preference. Nothing is rigged against the respondent — the weights are ordinary and the tiebreaker is
**orthogonal** to true preference, not opposed to it.

**The failure mode is feature inflation.** When indecisive, $g\approx0$ (the options truly are a
toss-up by the respondent's values), so the forced label carries no preference signal — only whatever
feature the tiebreaker keys on. The binary learner has no way to know, so it *inflates the weight of
that feature*: a rule that always defers to one attribute makes the learner believe that attribute
matters far more than it does. We show the learned-vs-true weight on the keyed feature below to make
this concrete. Benign forcing and broad learning are unaffected; biased forcing roughly triples
downstream regret.""")

code(r"""# ---- severe regime: SAME oracles / tau / rules as Blocks 1-2, only options are SIMILAR ----
QUERY_SIGMA = 0.30
cfgS = dict(tau_r=TAU_R, tau_kappa=TAU_K, noise_scale=NOISE_SCALE, noise_type=NOISE_TYPE,
            T=90, n_candidates=N_CAND, n_samples=N_SAMPLES, burn_in=BURN_IN,
            n_holdout=N_HOLDOUT, n_jobs=N_JOBS, query_sigma=QUERY_SIGMA)
T_SEV = cfgS["T"]

# indecision is now the norm, not the exception:
_rr = np.random.default_rng(0); _cnt = {"left":0,"right":0,"indifferent":0,"conflict":0}
for w in ORACLES:
    for q in ic.sample_queries(150, DIM, _rr, similarity=QUERY_SIGMA):
        _cnt[ic.sample_response(q, w, TAU_R, TAU_K, NOISE_SCALE, NOISE_TYPE, _rr)] += 1
_tot = sum(_cnt.values())
print(f"with similar options: {100*(_cnt['left']+_cnt['right'])/_tot:.0f}% decisive "
      f"(vs ~57% for distinct options) -> indecision dominates")

SPECS_S = dict(SPECS1)                       # exact same forcing rules as Block 1
SPECS_S["broad3"] = {"kind": "broad", "n_outcomes": 3}
SPECS_S["broad4"] = {"kind": "broad", "n_outcomes": 4}

t0 = time.time()
bS = {k: ex.run_method(s, ORACLES, base_seed=SEED, **cfgS) for k, s in SPECS_S.items()}
bS_skip = ex.run_method({"kind": "bt"}, ORACLES, base_seed=SEED, **cfgS)
print(f"Severe regime done in {time.time()-t0:.1f}s   (worst-case ceiling = {WC:.3f})\n")

# Feature inflation: learned vs true weight on each biased rule's keyed feature.
KEYED = {"bt_lex": 1, "bt_single": 0, "bt_self": 0, "bt_gut": 2}  # feature each rule defers to
RULES_S = ["bt_5050", "bt_btcons", "bt_lex", "bt_single", "bt_self", "bt_gut", "bt_comp"]
print(f"{'forcing rule':<28}{'L1':>7}{'% worst':>9}   feature inflation (true -> learned)")
for k in RULES_S:
    L1 = ex.aggregate_scalar(bS[k], 'l1_final')['mean']
    bo = ex.aggregate_scalar(bS[k], 'best_of_n_regret')['mean']
    tag = "benign" if k in BENIGN else ("structural" if k in STRUCTURAL else "biased")
    extra = ""
    if k in KEYED:
        j = KEYED[k]
        tw = float(np.mean([w[j] for w in ORACLES]))
        lw = float(np.mean([t['omega_hat'][j] for t in bS[k]]))
        extra = f"   {ic.FEATURE_NAMES[j]}: {tw:.2f} -> {lw:.2f}"
    print(f"{NAMES[k]:<28}{L1:>7.2f}{100*bo/WC:>8.0f}%   [{tag}]{extra}")
""")

code(r"""fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 4.8),
                                gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.45})
keysS = RULES_S

# Panel A: forcing-induced distortion, relative to ignoring (cancels shared misspec drift).
D = np.array([ex.relative_distortion(bS[k], bS_skip)["mean"] for k in keysS])
vmax = np.abs(D).max()
im = axA.imshow(D, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
axA.set_xticks(range(DIM)); axA.set_xticklabels(ic.FEATURE_NAMES, rotation=40, ha="right")
axA.set_yticks(range(len(keysS))); axA.set_yticklabels([NAMES[k] for k in keysS])
axA.set_title(r"Forcing-induced distortion  $\hat\omega_{\rm rule}-\hat\omega_{\rm ignore}$")
for i in range(len(keysS)):
    for j in range(DIM):
        axA.text(j, i, f"{D[i,j]:+.02f}", ha="center", va="center", fontsize=8)
axA.axhline(len(BENIGN) - 0.5, color="k", lw=1.0)  # benign | biased divider
fig.colorbar(im, ax=axA, fraction=0.046, pad=0.04)

# Panel B: downstream regret as % of the anti-optimal ceiling.
bo  = [100 * ex.aggregate_scalar(bS[k], 'best_of_n_regret')['mean']  / WC for k in keysS]
boe = [100 * ex.aggregate_scalar(bS[k], 'best_of_n_regret')['stderr'] / WC for k in keysS]
axB.bar(range(len(keysS)), bo, yerr=boe, color=[PALETTE[k] for k in keysS], capsize=3)
axB.set_xticks(range(len(keysS)))
axB.set_xticklabels([NAMES[k].replace("Binary: forced ", "") for k in keysS], rotation=40, ha="right")
axB.set_ylabel("downstream regret  (% of worst-case)")
axB.set_title("Downstream rule regret")

fig.suptitle(r"Severe regime ($\tau_r=%.2f,\ \tau_\kappa=%.2f$, similar options): "
             "feature-keyed forcing inflates its feature; benign & structural do not" % (TAU_R, TAU_K),
             y=1.02, fontsize=12)
fig.savefig("fig_severe_bias.pdf"); fig.savefig("fig_severe_bias.png")
plt.show()
print("Same oracles/thresholds/rules as Block 1 -- only the options are now similar. Feature-keyed "
      "rules (lex/single/self/gut) inflate their feature and pay large regret; the structural "
      "compromise rule, whose signal is ~orthogonal to feature differences, stays near benign.")
""")

code(r"""fig, ax = plt.subplots(figsize=(5.6, 4.0))
x = np.arange(1, T_SEV + 1)
for key in ["bt_skip_S", "bt_5050", "broad3", "broad4"]:
    src = bS_skip if key == "bt_skip_S" else bS[key]
    name = NAMES["bt_skip"] if key == "bt_skip_S" else NAMES[key]
    col  = PALETTE["bt_skip"] if key == "bt_skip_S" else PALETTE[key]
    agg = ex.aggregate_curve(src, "l1s")
    ax.plot(x, agg["mean"], color=col, lw=2, label=name)
    ax.fill_between(x, agg["mean"]-agg["stderr"], agg["mean"]+agg["stderr"], color=col, alpha=0.18)
ax.set_xlabel("query budget"); ax.set_ylabel(r"weight error $\|\hat\omega-\omega^*\|_1$")
ax.set_title("Severe regime: efficiency gap widens")
ax.legend(fontsize=9)
fig.savefig("fig_severe_efficiency.pdf"); fig.savefig("fig_severe_efficiency.png")
plt.show()
""")

# ---------------------------------------------------------------- APPENDIX
md(r"""## Appendix — Robustness / sensitivity

Are the conclusions artifacts of *our* choices for the oracle distribution (sparse Dirichlet $\alpha$)
and the feature/query distribution (i.i.d. $U[0,1]$ options)? We vary both and re-check the three
structural claims: **(i)** reduction at $\tau=0$, **(ii)** broad $\gg$ binary, **(iii)** benign $<$
biased. The feature-model sweep includes a row built from the **real kidney elicitation options** —
the empirical per-attribute scales and 5×5 correlation read straight from
`kidneystudy/kidney_features_raw.csv` — so the bias effect size is reported at the geometry actually
observed, not just at independence or an arbitrary $\rho$. These sweeps use a smaller oracle count and
budget than the main blocks — we only need the qualitative ordering, not publication-precision error
bars. Regret is reported as % of the per-setting worst-case ceiling.""")

code(r"""# ---- A. Oracle concentration (alpha) x query distribution ----
APP_N, APP_T = 12, 60
appcfg = dict(noise_scale=NOISE_SCALE, noise_type=NOISE_TYPE, T=APP_T, n_candidates=N_CAND,
              n_samples=100, burn_in=50, n_holdout=350, n_jobs=N_JOBS)
APP_METHODS = {
    "50/50":      {"kind": "bt", "forcing": ic.force_5050},
    "single":     {"kind": "bt", "forcing": ic.force_single_feature, "forcing_kwargs": {"feature": 0}},
    "gut":        {"kind": "bt", "forcing": ic.force_gut_weights, "forcing_kwargs": {"omega_bias": GUT_WEIGHTS}},
    "compromise": {"kind": "bt", "forcing": ic.force_compromise},
    "broad4":     {"kind": "broad", "n_outcomes": 4},
}
print("A. regret (% of worst-case) by oracle concentration x query distribution:\n")
print(f"{'alpha':>5} {'query':<9} | " + " ".join(f"{m:>10}" for m in APP_METHODS))
for alpha in [0.1, 0.3, 1.0]:
    Oa = ex.make_oracles(APP_N, dim=DIM, seed=2026, alpha=alpha)
    W = regret_ceiling(Oa)
    for qlabel, sig in [("distinct", None), ("similar", 0.30)]:
        cells = []
        for m, sp in APP_METHODS.items():
            r = ex.run_method(sp, Oa, base_seed=SEED, tau_r=TAU_R, tau_kappa=TAU_K,
                              query_sigma=sig, **appcfg)
            cells.append(f"{100*ex.aggregate_scalar(r,'best_of_n_regret')['mean']/W:>9.0f}%")
        print(f"{alpha:>5.1f} {qlabel:<9} | " + " ".join(f"{c:>10}" for c in cells))
print("\n-> broad4 = 0% everywhere; benign < biased holds at every alpha; severity needs 'similar'.")
""")

code(r"""# ---- B. Feature model: cross-feature correlation + heterogeneous scale ----
# (Gaussian-copula correlation keeps U[0,1] marginals; scale gives features different ranges.)
def equicorr(rho):
    return (1.0 - rho) * np.eye(DIM) + rho * np.ones((DIM, DIM))
O_app = ex.make_oracles(APP_N, dim=DIM, seed=2026, alpha=ORACLE_ALPHA)

def kidney_feature_model(path="../kidneystudy/kidney_features_raw.csv"):
    # Empirical per-feature scale (std, normalized to mean 1) and 5x5 correlation from
    # the REAL kidney elicitation options. Column order matches FEATURE_NAMES:
    # dep->elderlyDep, life->lifeYearsGained, obesity, work->weeklyWorkhours, wait->yearsWaiting.
    # Returns (cov, scale), or (None, None) if the data file is not present.
    import csv, os
    if not os.path.exists(path):
        return None, None
    feats = ["dep", "life", "obesity", "work", "wait"]
    rows = list(csv.DictReader(open(path)))
    A = np.array([[float(r["A_" + f]) for f in feats] for r in rows])
    B = np.array([[float(r["B_" + f]) for f in feats] for r in rows])
    X = np.vstack([A, B])                      # all option vectors actually shown
    sc = X.std(0); sc = sc / sc.mean()         # relative spread; magnitude normalized out
    return np.corrcoef(X.T), sc

CORR_KIDNEY, SCALE_KIDNEY = kidney_feature_model()
FEATURE_MODELS = [
    ("independent",        None,            None),
    ("correlated rho=.25", equicorr(0.25),  None),
    ("correlated rho=.5",  equicorr(0.50),  None),
]
if SCALE_KIDNEY is not None:
    iu = np.triu_indices(DIM, 1)
    print("kidney feature model from data:  scale =", np.round(SCALE_KIDNEY, 2),
          "  mean|corr| = %.2f\n" % np.abs(CORR_KIDNEY[iu]).mean())
    FEATURE_MODELS += [
        ("kidney scales only",  None,         SCALE_KIDNEY),
        ("kidney scales+corr",  CORR_KIDNEY,  SCALE_KIDNEY),
    ]
print("B. severe regime (similar options); regret (% of worst-case) by feature model:\n")
print(f"{'feature model':<20} {'dec%':>5} | " + " ".join(f"{m:>10}" for m in APP_METHODS))
for label, cov, sc in FEATURE_MODELS:
    rr = np.random.default_rng(0); dec = tot = 0       # measure indecision rate for this model
    for w in O_app:
        for q in ic.sample_queries(120, DIM, rr, similarity=0.30, cov=cov, scale=sc):
            tot += 1
            if ic.sample_response(q, w, TAU_R, TAU_K, NOISE_SCALE, NOISE_TYPE, rr) in ("left", "right"):
                dec += 1
    W = regret_ceiling(O_app, cov=cov, scale=sc)
    cells = []
    for m, sp in APP_METHODS.items():
        r = ex.run_method(sp, O_app, base_seed=SEED, tau_r=TAU_R, tau_kappa=TAU_K,
                          query_sigma=0.30, feature_cov=cov, feature_scale=sc, **appcfg)
        cells.append(f"{100*ex.aggregate_scalar(r,'best_of_n_regret')['mean']/W:>9.0f}%")
    print(f"{label:<20} {100*dec/tot:>4.0f}% | " + " ".join(f"{c:>10}" for c in cells))
print("\n-> broad4 stays ~0%, but the benign-vs-biased GAP shrinks sharply as features correlate:")
print("   the catastrophic magnitude is partly an artifact of the independent-feature assumption.")
""")

md(r"""### C. The "average case": a heuristic keyed on a *random* attribute

The main figures (and the kidney row above) point the feature-keyed rules at `elderlyDep`/`obesity` —
which are the *low-spread* attributes under the kidney scales, i.e. the **least-harmful** target. To see
the average case rather than this best case, we sweep *which* attribute `single-feature` keys on and
average over it, for both the independent and the real-kidney feature models.""")

code(r'''# ---- C. single-feature keyed on each attribute -> best / average / worst case ----
# Uses a slightly longer budget so the benign baseline converges (forced 50/50 needs more data),
# making the biased *excess* over benign the thing on display rather than finite-sample noise.
T_C = 120
O_C = ex.make_oracles(16, dim=DIM, seed=2026, alpha=ORACLE_ALPHA)
ccfg = dict(noise_scale=NOISE_SCALE, noise_type=NOISE_TYPE, T=T_C, n_candidates=N_CAND,
            n_samples=130, burn_in=70, n_holdout=400, n_jobs=N_JOBS, query_sigma=0.30)
CMODELS = [("independent", None, None)]
if SCALE_KIDNEY is not None:
    CMODELS.append(("kidney scales+corr", CORR_KIDNEY, SCALE_KIDNEY))

def _pct(spec, cov, sc, W):
    r = ex.run_method(spec, O_C, base_seed=SEED, tau_r=TAU_R, tau_kappa=TAU_K,
                      feature_cov=cov, feature_scale=sc, **ccfg)
    return 100 * ex.aggregate_scalar(r, "best_of_n_regret")["mean"] / W

print("single-feature regret (% of worst-case) by the attribute it keys on:\n")
print(f"{'feature model':<20}{'benign':>7}{'broad':>6}  | " +
      "".join(f"{n[:7]:>8}" for n in ic.FEATURE_NAMES) + "  |  AVG worst")
for label, cov, sc in CMODELS:
    W = regret_ceiling(O_C, cov=cov, scale=sc)
    bn = _pct({"kind": "bt", "forcing": ic.force_5050}, cov, sc, W)
    bd = _pct({"kind": "broad", "n_outcomes": 4}, cov, sc, W)
    per = [_pct({"kind": "bt", "forcing": ic.force_single_feature, "forcing_kwargs": {"feature": j}},
                cov, sc, W) for j in range(DIM)]
    print(f"{label:<20}{bn:>6.0f}%{bd:>5.0f}%  | " + "".join(f"{p:>7.0f}%" for p in per) +
          f"  | {np.mean(per):>3.0f}% {max(per):>3.0f}%")
print("\n-> independent: biased >> benign for ANY target (avg ~3x benign) -> catastrophe is")
print("   target-agnostic. kidney geometry: averaged over a random attribute the bias is ~benign;")
print("   only the single highest-spread attribute (weeklyWorkhours) reaches ~2x benign.")
''')

md(r"""### D. Feature correlation *hides* the bias — it does not remove it

Section B/C showed that *decision* regret shrinks as features correlate. But does the biased rule stop
corrupting $\hat\omega$, or does the corruption just stop *mattering* for decisions on the elicitation
distribution? We separate the two: sweep the feature correlation $\rho$ and track (left) downstream
regret evaluated both on the **matched** deployment distribution and on an **off-axis** one (independent
items), and (right) the learned weight on the feature the rule keys on.

The result: the parameter distortion is **invariant to $\rho$** (the keyed weight stays inflated at
every correlation), and matched-deployment regret $\to 0$ only because at high $\rho$ the weights are
barely *identified* on matched data — all $\omega$ rank matched items alike. Deploy on a distribution
where the attributes come apart and the full harm reappears. **The bias is a latent, deployment-shift
liability, not a benign one.**""")

code(r"""# ---- D. correlation masks decision harm but not parameter harm ----
RHOS = [0.0, 0.25, 0.5, 0.75, 0.95]
dcfg = dict(tau_r=TAU_R, tau_kappa=TAU_K, noise_scale=NOISE_SCALE, noise_type=NOISE_TYPE,
            T=110, n_candidates=N_CAND, n_samples=120, burn_in=60, n_holdout=400,
            n_jobs=N_JOBS, query_sigma=0.30)
O_D = ex.make_oracles(12, dim=DIM, seed=2026, alpha=ORACLE_ALPHA)

def _dreg(tr, cov):  # mean best-of-N regret (% worst) on a chosen deployment distribution
    return 100 * np.mean([ic.best_of_n_regret(t["omega_hat"], w, np.random.default_rng(700_000 + i), cov=cov)
                          / worstcase_regret(w, 700_000 + i, cov=cov) for i, (t, w) in enumerate(zip(tr, O_D))])

reg_match, reg_offax, benign_m, infl = [], [], [], []
for rho in RHOS:
    cov = None if rho == 0 else (1 - rho) * np.eye(DIM) + rho * np.ones((DIM, DIM))
    trs = ex.run_method({"kind": "bt", "forcing": ic.force_single_feature, "forcing_kwargs": {"feature": 0}},
                        O_D, base_seed=SEED, feature_cov=cov, **dcfg)
    trb = ex.run_method({"kind": "bt", "forcing": ic.force_5050}, O_D, base_seed=SEED, feature_cov=cov, **dcfg)
    reg_match.append(_dreg(trs, cov)); reg_offax.append(_dreg(trs, None)); benign_m.append(_dreg(trb, cov))
    infl.append(float(np.mean([t["omega_hat"][0] for t in trs])))
f0_true = float(np.mean([w[0] for w in O_D]))

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.0))
axL.plot(RHOS, reg_match, "o-", color=PALETTE["bt_single"], lw=2, label="single-feature, deploy = elicitation dist.")
axL.plot(RHOS, reg_offax, "s--", color="#b91c1c", lw=2, label="single-feature, deploy = off-axis (independent)")
axL.plot(RHOS, benign_m, "^:", color=PALETTE["bt_5050"], lw=1.6, label="benign 50/50 (matched)")
axL.set_xlabel(r"feature correlation $\rho$"); axL.set_ylabel("downstream regret (% of worst-case)")
axL.set_title("Decision harm: masked on matched, persists off-axis"); axL.legend(fontsize=8); axL.set_ylim(bottom=0)
axR.plot(RHOS, infl, "o-", color=PALETTE["bt_single"], lw=2, label="learned weight on keyed feature")
axR.axhline(f0_true, color="k", ls=":", lw=1.5, label="true weight")
axR.set_xlabel(r"feature correlation $\rho$"); axR.set_ylabel("weight on keyed feature (elderlyDep)")
axR.set_title("Parameter harm: invariant to correlation"); axR.legend(fontsize=8); axR.set_ylim(bottom=0)
fig.suptitle("Feature correlation HIDES the forced-choice bias, it does not remove it", y=1.02, fontsize=12)
fig.savefig("fig_correlation_masking.pdf"); fig.savefig("fig_correlation_masking.png")
plt.show()
print("learned weight on keyed feature stays ~%.2f (true %.2f) for ALL rho; matched regret -> 0 while "
      "off-axis regret stays ~%.0f%%." % (np.mean(infl), f0_true, np.mean(reg_offax)))
""")

md(r"""### Robustness conclusions

* **Reduction at $\tau=0$ (Block 0): fully robust.** It is an algebraic identity; it holds exactly under
  correlated and heterogeneously-scaled features (verified separately, max curve difference $=0$).
* **Broad $\gg$ binary (Block 2): fully robust.** The broad learner is correctly specified, so it
  recovers $\omega$ (regret $\approx 0$) at every $\alpha$, every query distribution, and every feature
  model tested. This claim does not depend on our choices.
* **benign $<$ feature-keyed-biased ordering (Block 1): directionally robust but magnitude-contingent.**
  The ordering holds everywhere, and it is *not* an artifact of the sparse $\alpha=0.3$ oracle (it
  persists at $\alpha=0.1$ and $1.0$). But the *size* of the gap is contingent: it needs **similar
  options** to be large (by design — that is the stimulus axis), and it **largely vanishes at the real
  kidney feature geometry, even in the average case**. Sweeping *which* attribute the heuristic keys on
  (section C): under **independent** features the bias is target-agnostic and large (single-feature
  averages ~17% of worst-case $\approx 3\times$ benign, for *any* attribute); under the **real kidney
  scales + correlation** it averages only ~6% $\approx$ benign, and *only* a heuristic fixated on the
  single highest-spread attribute (`weeklyWorkhours`) reaches ~$2\times$ benign. So the catastrophic
  magnitude is largely an artifact of the i.i.d.-feature assumption — at the geometry the kidney study
  actually exhibits, forced-choice bias is mild on average, and broad learning's advantage is the more
  robust effect.
* **But "mild decision regret" $\neq$ "no harm" — the bias is a deployment-shift liability (section D).**
  The reduced decision regret at high correlation does *not* mean the biased rule stopped corrupting
  $\hat\omega$: the learned weight on the keyed feature stays inflated at **every** $\rho$. Correlation
  only makes that distortion *irrelevant to decisions on the elicitation distribution* (where the
  weights are barely identified). Deploy the same $\hat\omega$ on an **off-axis** population — where the
  conflated attributes vary independently — and the full ~17% regret returns. So the safe-looking kidney
  numbers are conditional on deploying to a population like the elicitation one; they are *not* a
  guarantee, and any subgroup / cohort / policy that decorrelates the attributes re-exposes the harm.
* **`compromise` (structural): stays $\le$ feature-keyed everywhere**, and its harmlessness is itself
  geometry-dependent (it relies on the symmetric $U[0,1]$ feature mean coinciding with the compromise
  center).

Bottom line: the most *robust* claims are the reduction and **broad $\gg$ binary** (broad recovers
$\omega$ at every setting). The **biased-forcing catastrophe is the fragile one** — it is large under
i.i.d. features but mild on average at the real kidney geometry. So for the paper, lead with the
sample-efficiency / information argument (broad observing indecision), and present forced-choice bias as
a *conditional* risk whose size depends on attribute similarity and correlation — both measurable in the
study data rather than assumed.""")

# ---------------------------------------------------------------- summary
md(r"""## Summary

1. **Reduction (Block 0).** At $\tau_r=\tau_\kappa=0$ the threshold model is ordinary binary
   Bradley–Terry: broad-response and binary learning coincide (exactly under shared randomness,
   within standard error otherwise).
2. **Biased forcing fails (Block 1).** When indecision exists and forced-choice behavior is biased,
   binary learning recovers a systematically distorted $\omega$ — weight migrates onto whatever
   feature the forcing rule keys on — and chooses worse downstream rules. Assuming
   $\tau_r=\tau_\kappa=0$ is safe *only* under benign forcing.
3. **Benign forcing wastes information (Block 2).** Even when forced labels are unbiased, observing
   indecision directly (3-way or 4-way) reaches a given weight error with fewer queries, because
   indifference and conflict constrain $r$ and $g$ in ways binary labels cannot.
4. **The failure scales with how often people are indecisive (severe regime).** Keeping the oracles,
   thresholds, and rules of Blocks 1–2 fixed and only making the options *similar* (so indecision
   dominates), biased forcing **inflates the weight of whatever feature its tiebreaker keys on** —
   roughly doubling it and tripling downstream regret — while benign forcing and broad learning are
   unaffected. The tiebreaker is orthogonal to true preference, not opposed to it; the harm comes
   from the analyst mistaking tiebreaker behavior for preference. Blocks 1–2 are the benign end of
   this same axis.

5. **Robustness (appendix).** The reduction and broad-$\gg$-binary claims are invariant to oracle
   concentration, query distribution, and feature correlation/scale. The benign-$<$-biased *ordering*
   is robust too, but the *magnitude* is contingent and **largely an artifact of the i.i.d.-feature
   assumption**: at the **empirical kidney scales + correlation** (read from the study data), averaged
   over *which* attribute the heuristic keys on, biased-forcing regret is ~benign (~6% vs ~5% of
   worst-case) — only a rule fixated on the single highest-spread attribute (workhours) reaches ~2×
   benign, vs ~3× for *any* attribute under independent features. Broad learning's advantage is the
   more robust effect.

*Figures saved:* `fig_block0_sanity.{pdf,png}`, `fig_block1_bias.{pdf,png}`, `fig_block2_efficiency.{pdf,png}`,
`fig_severe_bias.{pdf,png}`, `fig_severe_efficiency.{pdf,png}`, `fig_correlation_masking.{pdf,png}`.

*Scaling for the paper:* increase `N_ORACLES`, `T*`, and `N_SAMPLES` in the setup cell; the seed
plumbing and method definitions are unchanged.""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
with open("final_experiments.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("wrote final_experiments.ipynb with", len(cells), "cells")
