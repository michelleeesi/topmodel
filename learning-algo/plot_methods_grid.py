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
    save_path=None,        # str
    show_stderr=True,      # bool
    figsize_per_cell=(1.5, 1.2),  # tuple
    title=None,            # str
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
        'l1': 'L1 Error',
        'holdout_ll': 'Holdout Log-Loss',
    }

    mean_key, stderr_key = metric_keys[metric]
    metric_title = metric_titles[metric]

    default_colors = {
        'multiframe': '#4477AA',
        'bt_laplace_bald': '#EE6677',
        'bt_laplace_bald_k': '#228833',
        'bt_laplace_bald_random': '#CCBB44',
        'bt_laplace_bald_left': '#AA3377',
    }
    default_labels = {
        'multiframe': 'Multi-frame',
        'bt_laplace_bald': 'BT (Skip)',
        'bt_laplace_bald_k': 'BT (K-Decisive)',
        'bt_laplace_bald_random': 'BT (Random FC)',
        'bt_laplace_bald_left': 'BT (Left FC)',
    }

    if method_colors is None:
        method_colors = default_colors
    if method_labels is None:
        method_labels = default_labels

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

                ax.plot(x, mean_curve, color=color, linewidth=1.5, label=label)
                if show_stderr and len(stderr_curve) == len(mean_curve):
                    ax.fill_between(x,
                        np.array(mean_curve) - np.array(stderr_curve),
                        np.array(mean_curve) + np.array(stderr_curve),
                        color=color, alpha=0.2)

            ax.set_xlim(1, T)
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.set_xlabel(f'τ={tau}', fontsize=9)
            if j == 0:
                ax.set_ylabel(f"τ'={tau_prime}", fontsize=9)

    axes[0, 0].legend(fontsize=7, loc='lower right')
    fig.text(0.5, 0.02, 'Queries', ha='center', fontsize=11)
    fig.text(0.02, 0.5, metric_title, va='center', rotation='vertical', fontsize=11)

    if title is None:
        title = f'{metric_title} Convergence'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved to {save_path}")
    plt.show()
    return fig
