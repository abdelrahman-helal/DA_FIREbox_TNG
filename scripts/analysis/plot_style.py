"""
Shared plotting style/helper for pred-vs-true halo-mass figures.

Standardizes what was previously duplicated (and inconsistent) inline across
notebooks: scatter color, marker size/alpha, and font size. Figures built from
a heteroscedastic (mean, variance) model use real per-node error bars from the
model's own predicted std, rather than a post-hoc binned/bootstrap estimate.
"""

import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 14, "axes.labelsize": 18, "axes.titlesize": 18, "legend.fontsize": 14})

SCATTER_COLOR = "#4C72B0"
MARKER_SIZE = 5
ELINEWIDTH = 0.6
ALPHA = 0.6


def plot_pred_vs_true_with_uncertainty(true, pred_mean, pred_std, ax, r2=None, rmse=None,
                                        title=None, color=None, annotate=None,
                                        annotate_fontsize=8, markersize=None):
    """
    Draw predicted (with per-node error bars from pred_std) vs. true halo mass on *ax*.

    The error bars are +/-1 * pred_std, i.e. one predicted standard deviation from the
    model's heteroscedastic head. They are NOT a 68% confidence interval: the
    moment-network loss (Jeffrey & Wandelt 2020) is likelihood-free -- it fits the first
    two moments and assumes no distributional family -- so +/-1 sigma only corresponds to
    68% coverage if the residuals happen to be Gaussian AND sigma is well calibrated.
    The companion pull histogram ((pred_mean - true) / pred_std, which should be N(0,1)
    when calibrated) is the diagnostic for exactly that; a reduced chi^2 above 1 means
    these bars are optimistic.

    Parameters
    ----------
    true, pred_mean, pred_std : array-like, 1-D, aligned per node.
    ax : matplotlib Axes to draw on.
    r2, rmse : optional metrics to fold into the title.
    title : optional title override; if None and r2/rmse given, one is built from them.
    color : optional marker/errorbar color; defaults to the module-level SCATTER_COLOR.
    annotate : optional text drawn INSIDE the axes (upper-left) instead of as a title.
        Use this when many panels share one figure and per-axes titles waste vertical
        space. Mutually usable with `title`, but normally you pass one or the other.
    annotate_fontsize : font size for the `annotate` text block.
    markersize : optional marker size override; defaults to module-level MARKER_SIZE.
    """
    c = SCATTER_COLOR if color is None else color
    ms = MARKER_SIZE if markersize is None else markersize

    ax.errorbar(
        true, pred_mean, yerr=pred_std,
        fmt="o", ecolor=c, mfc=c, mec="none",
        markersize=ms, elinewidth=ELINEWIDTH, alpha=ALPHA, capsize=0,
    )
    lo = min(min(true), min(pred_mean)) - 0.2
    hi = max(max(true), max(pred_mean)) + 0.2
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8)
    ax.grid(alpha=0.3)

    if title is not None:
        ax.set_title(title)
    elif r2 is not None and rmse is not None:
        ax.set_title(f"R$^2$={r2:.3f}  RMSE={rmse:.3f}")

    if annotate is not None:
        ax.text(
            0.03, 0.97, annotate, transform=ax.transAxes,
            va="top", ha="left", fontsize=annotate_fontsize,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.72),
            zorder=5,
        )

    return ax
