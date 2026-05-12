# Two-Agent Agreement-Loss under Diametric Opposition — Chart Documentation

This document walks through everything required to produce the three-panel chart
in §13b of `agreementset.ipynb` (the diametric-opposition / negative-weights
case). The chart shows:

1. **Left** — mean agreement loss of the LS-RUM minimax compromise, $L^\tau(F_{\rm LS})$.
2. **Middle** — mean agreement loss of the threshold-aware compromise, $L^\tau(F_\tau)$.
3. **Right** — improvement, $L^\tau(F_{\rm LS}) - L^\tau(F_\tau)$.

Each panel is a heatmap over the threshold grid $(\tau_\kappa, \tau_r)$.

---

## 1. Setting

We model two **agents** who have to agree on a single deterministic decision
**rule**. Each rule maps every input $x$ to a binary output $y \in \{0, 1\}$.
Both agents observe the same finite set of inputs and the same candidate
rule class; they differ in how they value the rule's behaviour.

### 1.1 Inputs

- $n_\text{inputs}$ inputs ($= 500$ by default). Each input $x$ presents a binary
  choice between $y = 0$ and $y = 1$.
- Each $(x, y)$ has a feature vector $\psi(x, y) \in [0, 1]^d$ with $d = 5$
  priority dimensions.
- Features are drawn iid $\psi(x, y) \sim U[0, 1]^d$ per rep.

### 1.2 Candidate rule class

- A rule is parameterised by $\theta \in \mathbb{R}^d$: $F_\theta(x) = \arg\max_y \langle \theta, \psi(x, y)\rangle$.
- $K_\text{rules}$ rules ($= 1000$ by default), $\theta_k$ drawn uniformly from the
  **L2 unit sphere** $\{\theta : \|\theta\|_2 = 1\}$. Sampling on the sphere
  (not the simplex) is what allows negative coefficients — a rule with
  $\theta_j < 0$ actively prefers *lower* $\psi_j$ values.
- Implementation: `sample_rule_params_general` (Gaussian sample → L2-normalise).

### 1.3 Two agents

Each agent $i$ has a weight vector $\omega_i \in \mathbb{R}^d$ (NOT restricted
to the simplex — entries can be negative).

For the diametric-opposition chart we use the maximally adversarial pair:
$$
\omega_1 = (0.45, 0.25, 0.15, 0.10, 0.05),\qquad \omega_2 = -\omega_1.
$$
Both have $\|\omega_i\|_1 = 1$ (so loss magnitudes are comparable to the
simplex baseline). Every priority direction agent 1 likes, agent 2 dislikes
equally.

---

## 2. Priority utilities

For each rule $F$ and each priority $j$:
$$
u_j(F) \;=\; \frac{1}{n_\text{inputs}} \sum_x \psi_j(x, F(x)).
$$
$u_j(F) \in [0, 1]$. The vector $u(F) \in [0, 1]^d$ is what either agent
"sees" when evaluating the rule.

Implementation: `compute_priority_utils(psi, rule_outputs)`.

Agent $i$'s scalar utility is the linear score $U_i(F) = \langle \omega_i, u(F)\rangle$.

---

## 3. Pairwise comparison: $\kappa$ and $r$

For each pair of rules $(F_a, F_b)$ and each agent $i$, define
$$
\kappa_i(F_a, F_b) = \langle \omega_i, u(F_a) - u(F_b) \rangle,
\qquad
r_i(F_a, F_b) = \bigl\langle |\omega_i|,\; |u(F_a) - u(F_b)| \bigr\rangle.
$$
- $\kappa$ is the **signed** scalar gap: positive when $F_a$ scores higher.
- $r$ is the **weight-magnitude-weighted absolute disagreement**. The
  $|\omega_i|$ factor is the generalisation that makes $r$ well-defined when
  $\omega$ has negative entries. It reduces to $\langle \omega_i, |d| \rangle$
  on the simplex.
- Always $r \ge |\kappa|$, so $\kappa/r \in [-1, 1]$.

Implementation: `compute_pairwise_r_kappa_general(pu, omega)`.

---

## 4. Thresholded domination relation

Given thresholds $(\tau_r, \tau_\kappa)$, rule $F_a$ **decisively dominates**
rule $F_b$ for agent $i$ iff
$$
r_i(F_a, F_b) \;\ge\; \tau_r
\qquad\text{AND}\qquad
\kappa_i(F_a, F_b) \;>\; \tau_\kappa \cdot r_i(F_a, F_b).
$$
- The $r$ condition gates **relevance** — the rules must differ enough to be
  worth comparing.
- The $\kappa$ condition gates **conflict severity** — the signed gap must be
  a large enough fraction of the absolute disagreement.

At $(\tau_r, \tau_\kappa) = (0, 0)$ this reduces to strict LS-RUM dominance:
$F_a$ dominates $F_b$ iff $U_i(F_a) > U_i(F_b)$.

### Domination margin

For agent $i$ and a candidate $F$:
$$
D_i^\tau(F) \;=\; \max_{F' :\, r_i(F', F) \ge \tau_r}\;\bigl[\kappa_i(F', F) - \tau_\kappa\,r_i(F', F)\bigr]_+.
$$
$D_i^\tau(F) = 0$ iff no other rule decisively dominates $F$ for agent $i$.

Implementation: `domination_margin(r, kappa, tau_r, tau_kappa)`.

### Agreement loss

The **two-agent agreement loss** is
$$
L^\tau(F) \;=\; \max\bigl(D_1^\tau(F),\; D_2^\tau(F)\bigr).
$$
This is the quantity plotted in the heatmaps. Lower means $F$ is harder for
the worst agent to decisively reject.

---

## 5. Two compromise rules

Both rules are picked from the same finite candidate class
$\{F_{\theta_k}\}_{k=1}^{K_\text{rules}}$.

### 5.1 LS-RUM minimax compromise $F_{\rm LS}$

$$
F_{\rm LS} \;=\; \arg\min_F\; \max_{i \in \{1, 2\}}\,\bigl[U_i^\star - U_i(F)\bigr],
$$
where $U_i^\star = \max_F U_i(F)$ is agent $i$'s best achievable scalar
utility in the rule class. This is **independent of $(\tau_r, \tau_\kappa)$**
— it's the scalar-only compromise.

Implementation (one line in the rep loop):
```python
F_ls = int(np.argmin(np.maximum(s1.max() - s1, s2.max() - s2)))
```

### 5.2 Threshold-aware compromise $F_\tau$

$$
F_\tau \;=\; \arg\min_F\; L^\tau(F).
$$
This depends on $(\tau_r, \tau_\kappa)$. It is recomputed in every cell of the
threshold grid.

Implementation (per cell):
```python
D1 = domination_margin(r1, k1, tau_r, tau_kappa)
D2 = domination_margin(r2, k2, tau_r, tau_kappa)
F_tau = int(np.argmin(np.maximum(D1, D2)))
```

By construction $L^\tau(F_\tau) \le L^\tau(F_{\rm LS})$ at every cell, so the
improvement panel is non-negative.

---

## 6. Sweep and replication

To get a stable mean we average over many independent worlds (i.e., resamples
of $\psi$ and the rule class).

- **Threshold grid**: $\tau_r \in \{0, 0.05, 0.10, \dots, 0.50\}$ (11 values),
  $\tau_\kappa \in \{0, 0.1, 0.2, \dots, 1.0\}$ (11 values).
- **Reps**: $n_\text{reps} = 100$. Each rep uses seed $= \text{base\_seed} + \text{rep}$ for
  reproducibility.
- **Per rep, per cell**: record $L^\tau(F_{\rm LS})$ and $L^\tau(F_\tau)$ into
  arrays of shape `(n_reps, n_r, n_k)`.

Implementation: `run_diagnostics_general(DEFAULT_CONFIG, omega_1_neg, omega_2_neg)`
returns the dict `DIAG_NEG` containing keys `"loss_ls"`, `"loss_tau"`,
`"improvement"`, plus the threshold grids.

---

## 7. The chart itself

```python
ng_L_LS  = DIAG_NEG["loss_ls"].mean(0)        # (n_r, n_k)
ng_L_tau = DIAG_NEG["loss_tau"].mean(0)
ng_imp   = DIAG_NEG["improvement"].mean(0)    # >= 0 by construction
vmax_loss = max(ng_L_LS.max(), ng_L_tau.max())
extent_n  = [tau_kappa_min, tau_kappa_max, tau_r_min, tau_r_max]
```

**Layout**: `plt.subplots(1, 3, figsize=(15, 4.5))`.

**Per panel**:
- `origin='lower'` (so $\tau_r = 0$ is at the bottom).
- `aspect='auto'`, `extent=extent_n` so axes are labelled in true $(\tau_\kappa, \tau_r)$ coordinates.
- Left & middle panels share the colour scale `vmin=0, vmax=vmax_loss` (so the
  two losses are visually comparable). Cmap `'magma'`.
- Right panel uses `vmin=0, vmax=ng_imp.max()`, cmap `'viridis'` (different
  scale; improvement is a derived quantity).
- $x$-axis label `$\tau_\kappa$`, $y$-axis label `$\tau_r$`.

**Output**: saved to `figures/diag_neg_only.png`.

---

## 8. Reading the chart

- **Active region**: low $\tau_r$ (roughly $\tau_r < 0.2$). Above that no rule
  pair achieves enough $r$ to satisfy the relevance gate, so $D_i^\tau = 0$ for
  every rule (the heatmap goes black).
- **Left panel ($F_{\rm LS}$)**: brightest near the corner $(\tau_r, \tau_\kappa) = (0, 0)$, where
  the LS rule still has a decisive scalar challenger (in fact, the LS optima
  of either agent).
- **Middle panel ($F_\tau$)**: dimmer than the left panel everywhere — the
  threshold-aware optimizer picks a different rule whose worst challenger is
  filtered out by the $\tau_r$ / $\tau_\kappa$ gates.
- **Right panel (improvement)**: the cells where the two strategies actually
  differ. Bright in a thin band along low $\tau_r$ and moderate $\tau_\kappa$.

The lower portion of `agreementset.ipynb` §13b also prints, for the cell with
the largest mean improvement, the actual $\theta$ vectors and priority-utility
vectors of $F_{\rm LS}$ and $F_\tau$ across the first three reps — useful for
seeing that the two rules are genuinely different policies (output-space
Hamming distance $\approx 0.5$, completely different $\theta$ sign patterns).

---

## 9. Dependencies and ordering

These cells/objects must be defined (and run) **before** §13b can execute:

| Earlier cell | What it defines | Used in §13b |
|---|---|---|
| §1 setup       | `DEFAULT_CONFIG`, `FIGDIR`, `plt`, `np` | yes |
| §2 rule class  | `apply_rules`, `sample_rule_params` (simplex, not strictly needed here) | indirectly |
| §3 priority utils | `compute_priority_utils` | yes |
| §13 helpers    | `sample_rule_params_general`, `compute_pairwise_r_kappa_general` | yes |
| §13 agents     | `omega_1_neg`, `omega_2_neg` | yes |
| §13a sweep     | `run_diagnostics_general`, `DIAG_NEG` | **required input** |
| §5 margin func | `domination_margin` | yes (in `_winning_rules_one_rep`) |
| §7 sweep grid  | `TAU_R_VALUES`, `TAU_KAPPA_VALUES` | yes |

In other words: run §1 through §13a end-to-end before plotting the chart.
At `DEFAULT_CONFIG` (`n_reps=100, K_rules=1000`) the §13a sweep takes
~5–10 minutes.

---

## 10. Assumptions worth challenging

The chart's qualitative shape rests on the following modelling choices.
Changing any of them may move (or remove) the visible advantage:

1. **Rule diversity.** Rules sampled from the L2 unit sphere. If rules are
   sampled from a low-dimensional manifold (e.g., simplex) the diametric setup
   is less interesting because the rule class can't move into the orthogonal
   half-space agent 2 prefers.

2. **Number of inputs $n_\text{inputs}$.** Larger $n_\text{inputs}$ concentrates
   $u(F)$ around its deterministic limit and shrinks pairwise $r$. The active
   $\tau_r$ band scales as $\sim 1/\sqrt{n_\text{inputs}}$. See §14b for the
   empirical sweep.

3. **Rule-space size $K_\text{rules}$.** Improvement is non-monotonic in $K$:
   peaks around $K \approx 200$ and shrinks beyond. With many rules both
   compromise rules find near-optimal candidates close to each other in
   $u$-space. See §14c.

4. **Agent norm.** Both agents have $\|\omega_i\|_1 = 1$. Rescaling either
   agent scales their kappa and r equally, so kappa/r is invariant — but loss
   magnitudes are not.

5. **Symmetry $\omega_2 = -\omega_1$.** Maximally adversarial; the LS-minimax
   rule sits at the midpoint of $U_1$ range and both losses are pinned to
   roughly $(U_1^\star - U_1^{\min}) / 2$. Asymmetric oppositions
   (e.g. $\omega_2 = -P\omega_1$ for some permutation $P$) give qualitatively
   similar but quantitatively different pictures.

6. **Linear score rules.** Each rule is $F_\theta(x) = \arg\max_y \langle \theta, \psi(x, y) \rangle$.
   Nonlinear classes (e.g., trees) would change the geometry of $u$-space
   and likely shift the active band.

7. **iid uniform features.** $\psi$ from $U[0, 1]^d$. Heavy-tailed or
   correlated feature distributions could enlarge pairwise $r$ and extend the
   $\tau_r$ axis.
