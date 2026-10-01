"""Reliability diagrams."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import calibration as cal


def reliability_panel(ax_rel, ax_hist, probs, labels, title, n_bins=15):
    """Equal-width bars (accuracy per bin) with the gap to the diagonal shaded,
    equal-mass bins overlaid as points, and a confidence histogram below."""
    w = cal.reliability_bins(probs, labels, n_bins, "width")
    m = cal.reliability_bins(probs, labels, n_bins, "mass")
    edges = w["edges"]
    width = np.diff(edges)
    has = w["count"] > 0

    ax_rel.plot([0, 1], [0, 1], ls="--", lw=1, color="0.4", label="perfect calibration")
    ax_rel.bar(edges[:-1][has], w["acc"][has], width=width[has], align="edge",
               color="#0F6E78", alpha=0.85, edgecolor="white", label="accuracy (equal-width bins)")
    gap_lo = np.minimum(w["acc"][has], w["conf"][has])
    gap_hi = np.maximum(w["acc"][has], w["conf"][has])
    ax_rel.bar(edges[:-1][has], gap_hi - gap_lo, bottom=gap_lo, width=width[has], align="edge",
               color="#C25F1A", alpha=0.35, edgecolor="none", label="gap to mean confidence")
    mh = m["count"] > 0
    ax_rel.plot(m["conf"][mh], m["acc"][mh], "o-", ms=4, lw=1, color="#17202E",
                label="equal-mass bins")
    ece_w = cal.ece(probs, labels, n_bins, "width")
    ece_m = cal.ece(probs, labels, n_bins, "mass")
    ax_rel.set_title(f"{title}\nacc {cal.accuracy(probs, labels):.3f} · ECE {ece_w:.3f} (width) / "
                     f"{ece_m:.3f} (mass) · Brier {cal.brier(probs, labels):.3f}", fontsize=9)
    ax_rel.set_xlim(0, 1)
    ax_rel.set_ylim(0, 1)
    ax_rel.set_ylabel("accuracy")
    ax_rel.legend(fontsize=7, loc="upper left")

    conf = probs.max(1)
    ax_hist.hist(conf, bins=edges, color="0.55", edgecolor="white")
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yscale("log")
    ax_hist.set_xlabel(f"top-label confidence ({n_bins} equal-width bins)")
    ax_hist.set_ylabel("count (log)")


def reliability_figure(panels: list[tuple[str, np.ndarray]], labels: np.ndarray, path, n_bins=15):
    fig, axes = plt.subplots(2, len(panels), figsize=(5.2 * len(panels), 6.4),
                             gridspec_kw={"height_ratios": [3, 1]}, squeeze=False)
    for j, (title, probs) in enumerate(panels):
        reliability_panel(axes[0, j], axes[1, j], probs, labels, title, n_bins)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
