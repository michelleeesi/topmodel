"""
Simulation: Inseparable Distributional Frame Confounds Score-RUM Learning
=========================================================================
Housing allocation, large X.

X = 800 inputs, sampled once and fixed.
Features (D=3):
  psi_1: need category in {0,1,2}, recoded to (2-raw)/2 so 1=high need
  psi_2: wait time in [0,1]
  psi_3: income fit in [0,1]

m=4 priorities:
  j=1,2,3 separable, omega_j = beta*_j*(1-omega_C), beta*=(0.5,0.3,0.2)
  j=C     inseparable: u_C(F) = -sum_g (p_g^F - alpha_g)^2
          alpha=(0.2,0.3,0.5): wants 50% low-need (G3), opposing true priorities

phi^rules = max over N_BG background rules.

Welfare metric: weighted disagreement fraction
  WDF = (1/|X|) * sum_x 1[learner picks wrong y at x] * |<omega_sep, delta(x)>|
      / (1/|X|) * sum_x |<omega_sep, delta(x)>|
  = fraction of separable utility left on table by learner's rule.
  This is in [0,1]: 0=perfect, 1=worst possible.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.special import expit
from scipy.optimize import minimize

# ── Parameters ────────────────────────────────────────────────────────────────
D              = 3
BETA_STAR      = np.array([0.5, 0.3, 0.2])
ALPHA_C        = np.array([0.2, 0.3, 0.5])
N_INPUTS       = 800
N_QUERIES      = 3000
N_RUNS         = 60
LOGISTIC_SCALE = 0.3
QUERY_BATCH    = 60
OMEGA_C_VALUES = [0.0, 0.2, 0.4, 0.6]
N_BG           = 2000   # background rules for inseparable evidence

# ── Feature sampling (fixed once per run) ────────────────────────────────────
def sample_features(rng):
    psi          = np.zeros((N_INPUTS, 2, D))
    psi[:, :, 0] = rng.integers(0, 3, size=(N_INPUTS, 2)).astype(float)
    psi[:, :, 1] = rng.uniform(0, 1, size=(N_INPUTS, 2))
    psi[:, :, 2] = rng.uniform(0, 1, size=(N_INPUTS, 2))
    # recode need: (2-raw)/2 so 1=high need, 0=low need
    psi[:, :, 0] = (2 - psi[:, :, 0]) / 2.0
    return psi  # (N_INPUTS, 2, D)

def sep_gap(psi):
    """delta(x) = psi(x,y=1) - psi(x,y=0), shape (N_INPUTS, D)."""
    return psi[:, 1, :] - psi[:, 0, :]   # (N_INPUTS, D)

# ── Inseparable evidence (precomputed once per instance) ──────────────────────
def compute_s_C(psi_raw, rng):
    """
    psi_raw: original (un-recoded) features so group is in {0,1,2}.
    For each input x:
      s_C_plus[x]  = max_F [u_C(F_{x->1}) - u_C(F_{x->0})]^+
      s_C_minus[x] = max_F [u_C(F_{x->0}) - u_C(F_{x->1})]^+
    Background rules: N_BG random binary vectors over N_INPUTS inputs.
    """
    # Sample background rules
    bg       = rng.integers(0, 2, size=(N_BG, N_INPUTS))   # (N_BG, N_INPUTS)
    # chosen_g[r, x] = group of applicant chosen by rule r at input x
    # = psi_raw[x, bg[r,x], 0]
    # Use advanced indexing: rows=input indices broadcast, cols=bg choices
    x_idx    = np.arange(N_INPUTS)[None, :]               # (1, N_INPUTS)
    chosen_g = psi_raw[x_idx, bg, 0].astype(int)          # (N_BG, N_INPUTS)

    # bar_p[r,g] = fraction of inputs where rule r allocates to group g
    bar_p = np.zeros((N_BG, 3))
    for g in range(3):
        bar_p[:, g] = (chosen_g == g).mean(axis=1)

    # group of y=0 and y=1 applicants at each input
    gA = psi_raw[:, 0, 0].astype(int)   # (N_INPUTS,) group of y=0 applicant
    gB = psi_raw[:, 1, 0].astype(int)   # (N_INPUTS,) group of y=1 applicant

    sp = np.zeros(N_INPUTS)
    sm = np.zeros(N_INPUTS)

    for x in range(N_INPUTS):
        gc = chosen_g[:, x]   # (N_BG,) group currently assigned at x by each bg rule

        # bar_p after setting F(x)=y=1 (gB[x]) vs F(x)=y=0 (gA[x])
        bB = bar_p.copy()
        bB[np.arange(N_BG), gc]     -= 1.0 / N_INPUTS
        bB[np.arange(N_BG), gB[x]] += 1.0 / N_INPUTS

        bA = bar_p.copy()
        bA[np.arange(N_BG), gc]     -= 1.0 / N_INPUTS
        bA[np.arange(N_BG), gA[x]] += 1.0 / N_INPUTS

        # u_C = -sum_g (bar_p_g - alpha_g)^2
        uCB = -np.sum((bB - ALPHA_C)**2, axis=1)
        uCA = -np.sum((bA - ALPHA_C)**2, axis=1)
        d   = uCB - uCA   # (N_BG,): Delta^F_C(y=1,y=0; x) for each bg rule

        sp[x] = max(d.max(),  0)
        sm[x] = max((-d).max(), 0)

    return sp, sm

# ── Response generation ───────────────────────────────────────────────────────
def generate_responses(psi, xs, omega_sep, omega_C, sp, sm, rng):
    d      = sep_gap(psi)[xs]                          # (T, D)
    g_sep  = (omega_sep * np.maximum(d, 0)).sum(1) \
           - (omega_sep * np.maximum(-d, 0)).sum(1)   # (T,)
    g_C    = omega_C * (sp[xs] - sm[xs])               # (T,)
    g      = g_sep + g_C
    prob   = expit(g / LOGISTIC_SCALE)
    return np.where(rng.uniform(size=len(xs)) < prob, 1, -1)

# ── Score-RUM MLE ─────────────────────────────────────────────────────────────
def fit_score_rum(deltas, responses):
    labels = (responses + 1) / 2
    def obj(raw):
        b   = np.exp(raw);  b /= b.sum()
        g   = (deltas @ b) / LOGISTIC_SCALE
        p   = expit(g)
        ll  = (labels * np.log(p+1e-15) + (1-labels)*np.log(1-p+1e-15)).mean()
        res = p - labels
        gb  = (res[:, None] * deltas).mean(0) / LOGISTIC_SCALE
        J   = np.diag(b) - np.outer(b, b)
        return -ll, J @ gb
    r = minimize(obj, np.zeros(D), jac=True, method='L-BFGS-B',
                 options={'maxiter': 600, 'ftol': 1e-13})
    b = np.exp(r.x);  b /= b.sum()
    return b

# ── Welfare metric ────────────────────────────────────────────────────────────
def weighted_disagreement(psi, hat_omega_scaled, omega_sep):
    """
    For each input x, compare the greedy choice under hat_omega_scaled
    vs the true greedy choice under omega_sep.
    Returns weighted disagreement fraction in [0,1]:
      WDF = sum_{x: disagree} |<omega_sep, delta(x)>|
          / sum_x |<omega_sep, delta(x)>|
    """
    d        = sep_gap(psi)                                # (N_INPUTS, D)
    score_true = d @ omega_sep                             # (N_INPUTS,)
    score_hat  = d @ hat_omega_scaled                      # (N_INPUTS,)
    true_y1    = score_true > 0                            # (N_INPUTS,) bool
    hat_y1     = score_hat  > 0                            # (N_INPUTS,) bool
    disagree   = true_y1 != hat_y1                         # (N_INPUTS,) bool
    gap        = np.abs(score_true)                        # utility gap at each x
    denom      = gap.sum()
    if denom < 1e-10:
        return 0.0
    return gap[disagree].sum() / denom

# ── Single run ────────────────────────────────────────────────────────────────
def run_one(omega_C, psi, psi_raw, sp, sm, rng):
    omega_sep   = BETA_STAR * (1 - omega_C)
    xs          = rng.integers(0, N_INPUTS, size=N_QUERIES)
    responses   = generate_responses(psi, xs, omega_sep, omega_C, sp, sm, rng)
    deltas      = sep_gap(psi)[xs]                         # (N_QUERIES, D)
    checkpoints = list(range(QUERY_BATCH, N_QUERIES+1, QUERY_BATCH))
    traj_omega  = np.zeros((len(checkpoints), D))
    traj_wdf    = np.zeros(len(checkpoints))
    for ci, cp in enumerate(checkpoints):
        ho              = fit_score_rum(deltas[:cp], responses[:cp])
        traj_omega[ci]  = ho
        traj_wdf[ci]    = weighted_disagreement(psi, ho * (1 - omega_C), omega_sep)
    return checkpoints, traj_omega, traj_wdf

# ── Main simulation ───────────────────────────────────────────────────────────
print("Running simulations...")
results = {}
for omega_C in OMEGA_C_VALUES:
    print(f"  omega_C = {omega_C:.2f}", flush=True)
    master_rng = np.random.default_rng(int(omega_C * 1000) + 42)
    trajs_o = []; trajs_w = []
    for run_i in range(N_RUNS):
        # Each run gets its own fixed feature instance
        run_rng  = np.random.default_rng(int(omega_C * 1000) + run_i * 17 + 1)
        psi_raw  = np.zeros((N_INPUTS, 2, D))
        psi_raw[:, :, 0] = run_rng.integers(0, 3, size=(N_INPUTS, 2)).astype(float)
        psi_raw[:, :, 1] = run_rng.uniform(0, 1, size=(N_INPUTS, 2))
        psi_raw[:, :, 2] = run_rng.uniform(0, 1, size=(N_INPUTS, 2))
        psi = psi_raw.copy()
        psi[:, :, 0] = (2 - psi_raw[:, :, 0]) / 2.0
        sp, sm = compute_s_C(psi_raw, run_rng)
        chk, to, tw = run_one(omega_C, psi, psi_raw, sp, sm, run_rng)
        trajs_o.append(to); trajs_w.append(tw)
    results[omega_C] = (chk, np.array(trajs_o), np.array(trajs_w))
print("Done.\n")

# ── MLE limits ────────────────────────────────────────────────────────────────
print("Estimating MLE limits (T=40000)...")
limits = {}
for omega_C in OMEGA_C_VALUES:
    lrng     = np.random.default_rng(99999)
    psi_raw  = np.zeros((N_INPUTS, 2, D))
    psi_raw[:, :, 0] = lrng.integers(0, 3, size=(N_INPUTS, 2)).astype(float)
    psi_raw[:, :, 1:] = lrng.uniform(0, 1, size=(N_INPUTS, 2, 2))
    psi_l    = psi_raw.copy(); psi_l[:,:,0] = (2-psi_raw[:,:,0])/2.0
    sp, sm   = compute_s_C(psi_raw, lrng)
    omega_sep = BETA_STAR * (1 - omega_C)
    xs_l     = lrng.integers(0, N_INPUTS, size=40000)
    resp_l   = generate_responses(psi_l, xs_l, omega_sep, omega_C, sp, sm, lrng)
    dlts_l   = sep_gap(psi_l)[xs_l]
    lim      = fit_score_rum(dlts_l, resp_l)
    limits[omega_C] = lim
    print(f"  omega_C={omega_C:.2f}: limit = {lim.round(4)}  (true beta* = {BETA_STAR})")

# ── Plot ──────────────────────────────────────────────────────────────────────
FEAT_NAMES  = [r"$\omega_1$: need", r"$\omega_2$: wait time", r"$\omega_3$: income fit"]
FEAT_COLORS = ["#e05c5c", "#4a90d9", "#52b788"]
GOLD        = "#f0a500"

fig = plt.figure(figsize=(16, 8))
fig.patch.set_facecolor('#0d0d14')
gs  = gridspec.GridSpec(2, 4, hspace=0.50, wspace=0.30,
                        left=0.08, right=0.97, top=0.87, bottom=0.09)

for col, omega_C in enumerate(OMEGA_C_VALUES):
    chk, all_o, _ = results[omega_C]
    mean_o = all_o.mean(0);  std_o = all_o.std(0)
    xs_plt = np.array(chk) / N_QUERIES
    lim    = limits[omega_C]

    # row 0: weight trajectories
    ax = fig.add_subplot(gs[0, col])
    ax.set_facecolor('#181824')
    for sp in ax.spines.values(): sp.set_color('#2a2a3a')
    for j in range(D):
        ax.plot(xs_plt, mean_o[:, j], color=FEAT_COLORS[j], lw=2.0)
        ax.fill_between(xs_plt,
                        mean_o[:, j] - std_o[:, j],
                        mean_o[:, j] + std_o[:, j],
                        color=FEAT_COLORS[j], alpha=0.15)
        ax.axhline(BETA_STAR[j], color=FEAT_COLORS[j], ls='--', lw=1.1, alpha=0.5)
        ax.axhline(lim[j],       color=FEAT_COLORS[j], ls=':',  lw=1.7, alpha=0.92)
    ax.set_ylim(-0.05, 1.05);  ax.set_xlim(0, 1)
    ax.tick_params(colors='#9999bb', labelsize=7.5)
    ax.set_title(rf"$\omega_C = {omega_C}$", color='#dde0f5',
                 fontsize=11, fontweight='bold', pad=5)
    if col == 0:
        ax.set_ylabel(r"learned $\hat\omega_j$", color='#9999bb', fontsize=9)
    ax.set_xlabel("fraction of queries", color='#9999bb', fontsize=8)

    # row 1: L1 error ||hat_omega - beta*||_1
    ax3 = fig.add_subplot(gs[1, col])
    ax3.set_facecolor('#181824')
    for sp in ax3.spines.values(): sp.set_color('#2a2a3a')

    l1_runs = np.abs(all_o - BETA_STAR[None, None, :]).sum(axis=2)
    mean_l1 = l1_runs.mean(0)
    std_l1  = l1_runs.std(0)
    lim_l1  = np.abs(lim - BETA_STAR).sum()

    ax3.fill_between(xs_plt,
                     np.maximum(mean_l1 - std_l1, 0),
                     mean_l1 + std_l1,
                     color='#a78bfa', alpha=0.18)
    ax3.plot(xs_plt, mean_l1, color='#a78bfa', lw=2.0)
    ax3.axhline(lim_l1, color='#a78bfa', ls=':', lw=1.7, alpha=0.92)

    ax3.set_ylim(0, None);  ax3.set_xlim(0, 1)
    ax3.tick_params(colors='#9999bb', labelsize=7.5)
    if col == 0:
        ax3.set_ylabel(r"$\ell_1$ error $\|\hat\omega - \beta^*\|_1$",
                       color='#9999bb', fontsize=9)
    ax3.set_xlabel("fraction of queries", color='#9999bb', fontsize=8)
    ax3.annotate(f"limit={lim_l1:.3f}",
                 xy=(0.97, 0.97), xycoords='axes fraction',
                 ha='right', va='top', fontsize=7.5, color='#a78bfa')

# legend
from matplotlib.lines import Line2D
handles = [Line2D([0],[0], color=FEAT_COLORS[j], lw=2.0,
                  label=f"{FEAT_NAMES[j]}  ($\\beta^*={BETA_STAR[j]}$)")
           for j in range(D)]
handles += [
    Line2D([0],[0], color='#aaaacc', lw=1.2, ls='--', label=r'true $\beta^*_j$'),
    Line2D([0],[0], color='#aaaacc', lw=1.7, ls=':',  label=r'MLE limit ($T\!\to\!\infty$)'),
    Line2D([0],[0], color='#a78bfa', lw=2.0, label=r'$\ell_1$ error (mean $\pm$ std)'),
]
fig.legend(handles=handles, loc='upper center', ncol=4, fontsize=8.5,
           frameon=False, labelcolor='#e0e0f5', bbox_to_anchor=(0.5, 0.975))

fig.suptitle(
    "Inseparable distributional frame confounds Score-RUM: housing allocation\n"
    r"$m\!=\!4$: need/wait/income ($\beta^*\!=\!(0.5,0.3,0.2)$) $+\ \omega_C$ on "
    r"$u_C(F)\!=\!-\!\sum_g(\bar p_g^F\!-\!\alpha_g)^2$, "
    r"$\alpha\!=\!(0.2,0.3,0.5)$"
    rf", $|X|\!=\!{N_INPUTS}$, $\phi^\mathrm{{rules}}\!=\!\max$",
    color='#eeeeff', fontsize=10, y=1.02, fontweight='bold'
)

outpath = "/mnt/user-data/outputs/confounding_simulation.png"
plt.savefig(outpath, dpi=160, bbox_inches='tight', facecolor='#0d0d14')
print(f"\nSaved to {outpath}")
