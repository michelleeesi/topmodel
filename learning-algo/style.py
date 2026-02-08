"""
Shared plotting style for paper experiments.

Professional, publication-quality style for Economics and Computation papers.
Based on best practices from:
- https://github.com/jbmouret/matplotlib_for_papers
- https://github.com/garrettj403/SciencePlots
- https://allanchain.github.io/blog/post/mpl-paper-tips/
"""

import matplotlib
import matplotlib.pyplot as plt
from cycler import cycler
from contextlib import contextmanager
import numpy as np

# ============================================================================
# UNIT CONVERSIONS
# ============================================================================
MM_TO_INCH = 1 / 25.4
PT_TO_INCH = 1 / 72

# Common paper column widths
SINGLE_COLUMN = 3.5  # inches (typical single column)
DOUBLE_COLUMN = 7.0  # inches (typical double column / full width)
NEURIPS_WIDTH = 5.5  # inches (NeurIPS text width)
ARXIV_WIDTH = 6.5    # inches


# ============================================================================
# COLOR PALETTES
# ============================================================================

# Colorblind-safe palette (Paul Tol's bright scheme)
COLORS_BRIGHT = [
    '#4477AA',  # blue
    '#EE6677',  # red/pink
    '#228833',  # green
    '#CCBB44',  # yellow
    '#66CCEE',  # cyan
    '#AA3377',  # purple
    '#BBBBBB',  # grey
]

# High contrast for presentations/posters
COLORS_HIGH_CONTRAST = [
    '#004488',  # dark blue
    '#BB5566',  # dark red
    '#DDAA33',  # gold
    '#000000',  # black
]

# Muted palette for dense plots
COLORS_MUTED = [
    '#332288',  # indigo
    '#88CCEE',  # cyan
    '#44AA99',  # teal
    '#117733',  # green
    '#999933',  # olive
    '#DDCC77',  # sand
    '#CC6677',  # rose
    '#882255',  # wine
    '#AA4499',  # purple
]

# Two-color comparison (e.g., method A vs method B)
COLOR_A = '#4477AA'  # blue (Bayesian)
COLOR_B = '#EE6677'  # red/coral (BT)

# Default palette
COLORS = COLORS_BRIGHT


# ============================================================================
# LINE STYLES
# ============================================================================

LINESTYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
MARKERS = ['o', 's', '^', 'D', 'v', 'p', 'h']


# ============================================================================
# STYLE SETUP
# ============================================================================

def setup_style(use_latex=True):
    """Apply publication-quality matplotlib style settings.

    Args:
        use_latex: If True, use LaTeX for text rendering (requires LaTeX installation).
                   This gives exact font matching with ACM/EC papers.
    """
    # Common style parameters for both modes
    common_params = {
        # Color cycle
        'axes.prop_cycle': cycler('color', COLORS),

        # Axes styling
        'axes.linewidth': 0.6,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.edgecolor': '#333333',
        'axes.labelcolor': '#333333',
        'axes.axisbelow': True,
        'axes.labelpad': 4,

        # Tick styling - inward ticks
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 3,
        'xtick.major.width': 0.6,
        'xtick.minor.size': 1.5,
        'xtick.minor.width': 0.4,
        'ytick.major.size': 3,
        'ytick.major.width': 0.6,
        'ytick.minor.size': 1.5,
        'ytick.minor.width': 0.4,
        'xtick.color': '#333333',
        'ytick.color': '#333333',
        'xtick.top': False,
        'ytick.right': False,

        # Grid - subtle
        'axes.grid': False,
        'grid.color': '#E0E0E0',
        'grid.linewidth': 0.4,
        'grid.alpha': 0.7,

        # Lines
        'lines.linewidth': 1.5,
        'lines.markersize': 5,
        'lines.markeredgewidth': 0.8,
        'lines.markeredgecolor': 'white',

        # Legend - no frame
        'legend.frameon': False,
        'legend.borderpad': 0.4,
        'legend.labelspacing': 0.3,
        'legend.handlelength': 1.5,
        'legend.handletextpad': 0.4,
        'legend.columnspacing': 1.0,
        'legend.borderaxespad': 0.5,

        # Figure
        'figure.dpi': 150,
        'figure.facecolor': 'white',
        'figure.edgecolor': 'white',
        'figure.constrained_layout.use': True,

        # Saving
        'savefig.dpi': 300,
        'savefig.format': 'pdf',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'white',

        # Patch (for fill_between, etc.)
        'patch.linewidth': 0.5,

        # Error bars
        'errorbar.capsize': 2,
    }

    if use_latex:
        # Use LaTeX rendering with libertine package for exact ACM font matching
        style_params = {
            'text.usetex': True,
            'text.latex.preamble': r'\usepackage{libertine}\usepackage[libertine]{newtxmath}',
            'font.family': 'serif',
            'font.size': 8,
            'axes.labelsize': 8,
            'axes.titlesize': 8,
            'legend.fontsize': 7,
            'xtick.labelsize': 7,
            'ytick.labelsize': 7,
        }
    else:
        # Fallback: use best available system fonts
        style_params = {
            'text.usetex': False,
            'font.family': 'serif',
            'font.serif': ['Palatino', 'Times New Roman', 'Times', 'DejaVu Serif'],
            'font.size': 8,
            'axes.labelsize': 8,
            'axes.titlesize': 8,
            'legend.fontsize': 7,
            'xtick.labelsize': 7,
            'ytick.labelsize': 7,
            'mathtext.fontset': 'stix',
        }

    # Merge common params with mode-specific params
    style_params.update(common_params)
    plt.rcParams.update(style_params)


# ============================================================================
# CONTEXT MANAGER
# ============================================================================

@contextmanager
def paper_style(
    figsize: tuple = None,
    width: float = SINGLE_COLUMN,
    aspect: float = 0.618,  # golden ratio
    use_latex: bool = True,
    font_size: int = 8,
):
    """
    Context manager for publication-quality plots.

    Usage:
        with paper_style(width=DOUBLE_COLUMN):
            fig, ax = plt.subplots()
            ax.plot(x, y)
            plt.savefig('figure.pdf')

    Args:
        figsize: Explicit (width, height) in inches. Overrides width/aspect.
        width: Figure width in inches
        aspect: Height/width ratio (default: golden ratio)
        use_latex: Use LaTeX rendering (requires LaTeX + libertine package)
        font_size: Base font size
    """
    if figsize is None:
        figsize = (width, width * aspect)

    # Build rcparams
    common_params = {
        # Color cycle
        'axes.prop_cycle': cycler('color', COLORS),

        # Figure size
        'figure.figsize': figsize,

        # Axes styling
        'axes.linewidth': 0.6,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.edgecolor': '#333333',
        'axes.labelcolor': '#333333',
        'axes.axisbelow': True,
        'axes.labelpad': 4,

        # Tick styling
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 3,
        'xtick.major.width': 0.6,
        'xtick.minor.size': 1.5,
        'xtick.minor.width': 0.4,
        'ytick.major.size': 3,
        'ytick.major.width': 0.6,
        'ytick.minor.size': 1.5,
        'ytick.minor.width': 0.4,
        'xtick.color': '#333333',
        'ytick.color': '#333333',
        'xtick.top': False,
        'ytick.right': False,

        # Grid
        'axes.grid': False,
        'grid.color': '#E0E0E0',
        'grid.linewidth': 0.4,
        'grid.alpha': 0.7,

        # Lines
        'lines.linewidth': 1.5,
        'lines.markersize': 5,
        'lines.markeredgewidth': 0.8,
        'lines.markeredgecolor': 'white',

        # Legend
        'legend.frameon': False,
        'legend.borderpad': 0.4,
        'legend.labelspacing': 0.3,
        'legend.handlelength': 1.5,
        'legend.handletextpad': 0.4,

        # Figure
        'figure.dpi': 150,
        'figure.facecolor': 'white',
        'figure.constrained_layout.use': True,

        # Saving
        'savefig.dpi': 300,
        'savefig.format': 'pdf',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,

        # Patch
        'patch.linewidth': 0.5,
        'errorbar.capsize': 2,
    }

    if use_latex:
        font_params = {
            'text.usetex': True,
            'text.latex.preamble': r'\usepackage{libertine}\usepackage[libertine]{newtxmath}',
            'font.family': 'serif',
            'font.size': font_size,
            'axes.labelsize': font_size,
            'axes.titlesize': font_size,
            'legend.fontsize': font_size - 1,
            'xtick.labelsize': font_size - 1,
            'ytick.labelsize': font_size - 1,
        }
    else:
        font_params = {
            'text.usetex': False,
            'font.family': 'serif',
            'font.serif': ['Palatino', 'Times New Roman', 'Times', 'DejaVu Serif'],
            'font.size': font_size,
            'axes.labelsize': font_size,
            'axes.titlesize': font_size,
            'legend.fontsize': font_size - 1,
            'xtick.labelsize': font_size - 1,
            'ytick.labelsize': font_size - 1,
            'mathtext.fontset': 'stix',
        }

    rcparams = {**common_params, **font_params}

    with matplotlib.rc_context(rcparams):
        yield


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def set_size(width: float = SINGLE_COLUMN, aspect: float = 0.618):
    """Get (width, height) tuple for figure size."""
    return (width, width * aspect)


def add_subplot_labels(
    axes,
    labels: list = None,
    x: float = -0.12,
    y: float = 1.08,
    fontweight: str = 'bold',
    fontsize: int = None,
):
    """
    Add (a), (b), (c), ... labels to subplots.

    Args:
        axes: List of axes or axes array from subplots
        labels: Custom labels (default: a, b, c, ...)
        x, y: Position relative to axes
        fontweight: Font weight for labels
        fontsize: Font size (default: axes title size)
    """
    axes_flat = np.array(axes).flatten()

    if labels is None:
        labels = [chr(ord('a') + i) for i in range(len(axes_flat))]

    for ax, label in zip(axes_flat, labels):
        ax.text(
            x, y, f'({label})',
            transform=ax.transAxes,
            fontweight=fontweight,
            fontsize=fontsize,
            va='bottom',
            ha='right',
        )


def despine(ax=None, top=True, right=True, left=False, bottom=False):
    """Remove axis spines."""
    if ax is None:
        ax = plt.gca()

    for spine, hide in [('top', top), ('right', right),
                         ('left', left), ('bottom', bottom)]:
        ax.spines[spine].set_visible(not hide)


def plot_with_band(
    ax, x, y_median, y_lower, y_upper,
    color=None, label=None, alpha=0.2, **kwargs
):
    """
    Plot line with shaded uncertainty band.

    Args:
        ax: Matplotlib axes
        x: x values
        y_median: Central line values
        y_lower: Lower bound of band
        y_upper: Upper bound of band
        color: Line/band color
        label: Legend label
        alpha: Band transparency
        **kwargs: Passed to ax.plot()
    """
    line, = ax.plot(x, y_median, color=color, label=label, **kwargs)
    color = line.get_color()
    ax.fill_between(x, y_lower, y_upper, color=color, alpha=alpha, linewidth=0)
    return line


def plot_comparison(
    ax, x, y_a, y_b,
    label_a='Method A', label_b='Method B',
    color_a=COLOR_A, color_b=COLOR_B,
    marker_a='o', marker_b='s',
    linestyle_a='-', linestyle_b='--',
    **kwargs
):
    """
    Plot two methods for comparison with consistent styling.

    Args:
        ax: Matplotlib axes
        x: x values
        y_a, y_b: y values for methods A and B
        label_a, label_b: Legend labels
        color_a, color_b: Colors
        marker_a, marker_b: Markers
        linestyle_a, linestyle_b: Line styles
        **kwargs: Passed to ax.plot()
    """
    ax.plot(x, y_a, color=color_a, marker=marker_a, linestyle=linestyle_a,
            label=label_a, **kwargs)
    ax.plot(x, y_b, color=color_b, marker=marker_b, linestyle=linestyle_b,
            label=label_b, **kwargs)


def savefig(fig, filename, formats=['pdf', 'png'], **kwargs):
    """
    Save figure in multiple formats.

    Args:
        fig: Matplotlib figure
        filename: Base filename (without extension)
        formats: List of formats to save
        **kwargs: Passed to fig.savefig()
    """
    for fmt in formats:
        fig.savefig(f'{filename}.{fmt}', format=fmt, **kwargs)


# ============================================================================
# QUICK START
# ============================================================================

if __name__ == '__main__':
    # Demo
    x = np.linspace(0, 10, 50)
    y1 = np.sin(x) + np.random.normal(0, 0.1, len(x))
    y2 = np.cos(x) + np.random.normal(0, 0.1, len(x))

    with paper_style(width=SINGLE_COLUMN, use_latex=False):
        fig, ax = plt.subplots()
        plot_comparison(ax, x, y1, y2, label_a='Method A', label_b='Method B')
        ax.set_xlabel('Sample Size')
        ax.set_ylabel('Error')
        ax.legend()
        plt.savefig('demo_plot.pdf')
        plt.show()

    print("Demo complete! Check demo_plot.pdf")
