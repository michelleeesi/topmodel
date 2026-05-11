"""
Function to plot multiple methods on the same convergence grid.
Copy this into cell 16 of the notebook.
"""

def plot_methods_grid(
    results,  # Dict[tuple, Dict[str, Dict]]
    taus,     # List[float]
    tau_primes,  # List[float]
    T,        # int
    methods_to_plot=None,  # List[str]
    metric='cos_sims',     # str: 'cos_sims', 'l1', 'holdout_ll'
    method_colors=None,    # Dict[str, str]
    method_labels=None,    # Dict[str, str]
    method_linestyles=None,  # Dict[str, str] — '-', '--', '-.', ':'
    save_path=None,        # str
    show_stderr=True,      # bool
    figsize_per_cell=(1.5, 1.2),  # tuple
    title=None,            # str — None to omit suptitle
    legend_ncol=4,         # int — columns in the figure-level bottom legend
):
    """
    Plot multiple methods on the same convergence grid.

    Example:
        plot_methods_grid(
            results, taus, tau_primes, T,
            methods_to_plot=['multiframe', 'bt_laplace_bald_k', 'bt_laplace_bald_random'],
            metric='cos_sims',
            save_path='mf_vs_k_vs_random.pdf'
        )
    """
    import matplotlib
    matplotlib.rcParams['text.usetex'] = False
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'

    metric_keys = {
        'cos_sims': ('mean', 'stderr'),
        'l1': ('l1_mean', 'l1_stderr'),
        'holdout_ll': ('holdout_ll_mean', 'holdout_ll_stderr'),
    }
    metric_titles = {
        'cos_sims': 'Cosine Similarity',
        'l1': r'$\|\hat\omega - \beta^*\|_1$',
        'holdout_ll': 'Holdout Log-Loss',
    }

    mean_key, stderr_key = metric_keys[metric]
    metric_title = metric_titles[metric]

    # Color families encode apples-to-apples groupings (Tol "Muted" + Wong
    # palettes — both validated for protanopia, deuteranopia, tritanopia):
    #   green-teal = noise-family known (Utilize-Indecision 4-out, 3-out)
    #   indigo-purple = no-noise-family knowledge (Utilize-Indecision 4-out + BT 2-out / MoG)
    #   warm = forced-choice variants (random, lex) — Wong vermillion + orange
    #   black = bt_hitandrun singleton
    default_colors = {
        # Greens — noise known Utilize-Indecision (Tol Muted)
        'multiframe':                '#117733',  # 4-out (dark teal-green)
        'multiframe_3outcome':       '#44AA99',  # 3-out (light teal-green)
        # Purples — no noise-family knowledge (Tol Muted)
        'multiframe_unknown_family': '#332288',  # 4-out (indigo)
        'bt_mog':                    '#AA4499',  # 2-out (purple)
        # Warms — forced-choice (Wong palette — vermillion + orange)
        'bt_hitandrun_random':       '#D55E00',  # vermillion (dark warm)
        'bt_hitandrun_lex':          '#E69F00',  # orange (light warm)
        # Singleton — Ignore-Indecision (logistic, scale matched)
        'bt_hitandrun':              '#000000',  # black (neutral, distinct from all families)
        # Legacy keys (kept for backward compatibility with old pkls) — also CB-safe
        'multiframe_unknown_noise':  '#999933',  # olive
        'bt_laplace_bald':           '#117733',  # share green family
        'bt_laplace_bald_k':         '#44AA99',
        'bt_laplace_bald_random':    '#D55E00',
        'bt_laplace_bald_left':      '#882255',  # wine
        'bt_laplace_bald_lex':       '#E69F00',
    }
    # Line styles encode base algorithm:
    #   solid  = Utilize-Indecision (uses indecisive responses)
    #   dotted = Ignore-Indecision  (drops indecisive responses)
    #   dashed = Force-Decision     (forces indecisive to a decision)
    default_linestyles = {
        'multiframe':                '-',
        'multiframe_3outcome':       '-',
        'multiframe_unknown_family': '-',
        'bt_hitandrun':              ':',
        'bt_mog':                    ':',
        'bt_hitandrun_random':       '--',
        'bt_hitandrun_lex':          '--',
        # Legacy keys
        'multiframe_unknown_noise':  '-',
        'bt_laplace_bald':           ':',
        'bt_laplace_bald_k':         ':',
        'bt_laplace_bald_random':    '--',
        'bt_laplace_bald_left':      '--',
        'bt_laplace_bald_lex':       '--',
    }
    default_labels = {
        'multiframe': 'Utilize-Indecision (4-outcome, noise known)',
        # 'multiframe_unknown_noise': 'Utilize-Indecision (kernel known, beta unknown)',
        'multiframe_unknown_family': 'Utilize-Indecision (no noise-family knowledge)',
        'multiframe_3outcome': 'Utilize-Indecision (3-outcome, noise known)',
        # 'bt_laplace_bald': 'Ignore-Indecision',
        # 'bt_laplace_bald_k': 'Ignore-Indecision-K',
        # 'bt_laplace_bald_random': 'Force-Decision (Random)',
        # 'bt_laplace_bald_left': 'BT (Left FC)',
        # 'bt_laplace_bald_lex': 'Force-Decision (Lexicographic)',
        'bt_mog': 'Ignore-Indecision (no noise-family knowledge)',
        'bt_hitandrun': 'Ignore-Indecision (noise known)',
        'bt_hitandrun_random': 'Force-Decision Random',
        'bt_hitandrun_lex': 'Force-Decision Lexicographic',
    }

    if method_colors is None:
        method_colors = default_colors
    if method_labels is None:
        method_labels = default_labels
    if method_linestyles is None:
        method_linestyles = default_linestyles

    sample_cell = list(results.values())[0]
    available = list(sample_cell.keys())

    if methods_to_plot is None:
        methods_to_plot = available
    else:
        methods_to_plot = [m for m in methods_to_plot if m in available]

    print(f"Plotting: {methods_to_plot}")

    n_rows, n_cols = len(tau_primes), len(taus)
    fig_w = figsize_per_cell[0] * n_cols + 2.0
    fig_h = figsize_per_cell[1] * n_rows + 1.5

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), sharex=True, sharey=True)

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for i, tau_prime in enumerate(tau_primes):
        for j, tau in enumerate(taus):
            ax = axes[n_rows - 1 - i, j]
            cell_results = results.get((tau, tau_prime), {})

            for method in methods_to_plot:
                if method not in cell_results:
                    continue
                data = cell_results[method]
                mean_curve = data.get(mean_key, [])
                stderr_curve = data.get(stderr_key, [])
                if len(mean_curve) == 0:
                    continue

                x = np.arange(1, len(mean_curve) + 1)
                color = method_colors.get(method, 'gray')
                label = method_labels.get(method, method)
                linestyle = method_linestyles.get(method, '-')

                ax.plot(x, mean_curve, color=color, linewidth=1.5,
                        linestyle=linestyle, label=label)
                if show_stderr and len(stderr_curve) == len(mean_curve):
                    ax.fill_between(x,
                        np.array(mean_curve) - np.array(stderr_curve),
                        np.array(mean_curve) + np.array(stderr_curve),
                        color=color, alpha=0.2)

            ax.set_xlim(1, T)
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.set_xlabel(rf'$\tau_r={tau}$', fontsize=9)
            if j == 0:
                ax.set_ylabel(rf'$\tau_\kappa={tau_prime}$', fontsize=9)

    # Pull handles/labels from any axes that has plotted lines (all axes
    # share the same set of methods, so any one with data works).
    handles, labels = [], []
    for row in axes:
        for ax in row:
            h, l = ax.get_legend_handles_labels()
            if l:
                handles, labels = h, l
                break
        if handles:
            break

    # Compute how tall the legend will be, in figure-fraction units, so we can
    # leave the right amount of space at the bottom of the grid.
    n_legend_rows = max(1, int(np.ceil(len(handles) / max(1, legend_ncol))))
    legend_height_frac = 0.025 * n_legend_rows           # ~0.025 per row
    queries_y          = 0.005                            # x-label sits at the bottom
    legend_anchor_y    = queries_y + 0.025                # legend sits just above x-label
    bottom_rect        = legend_anchor_y + legend_height_frac + 0.02

    fig.text(0.5, queries_y, 'Queries', ha='center', fontsize=11)
    fig.text(0.02, 0.5, metric_title, va='center', rotation='vertical', fontsize=11)

    fig.legend(handles, labels,
               loc='lower center', bbox_to_anchor=(0.5, legend_anchor_y),
               ncol=legend_ncol, frameon=False, fontsize=8,
               handlelength=2.5, columnspacing=1.3)

    if title is not None:
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0.03, bottom_rect, 1, 0.95])
    else:
        plt.tight_layout(rect=[0.03, bottom_rect, 1, 1])

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved to {save_path}")
    plt.show()
    return fig
