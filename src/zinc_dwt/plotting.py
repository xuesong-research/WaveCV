"""Plotting utilities."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional
import re

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, MaxNLocator

from .wavelet import decompose_wavelet_reconstructed
from .analysis import compute_first_differences

BASE_COLORS = [
    (45 / 255, 92 / 255, 135 / 255),
    (120 / 255, 153 / 255, 188 / 255),
    (247 / 255, 224 / 255, 165 / 255),
    (247 / 255, 213 / 255, 123 / 255),
    (210 / 255, 159 / 255, 200 / 255),
    (140 / 255, 130 / 255, 190 / 255),
]


def set_mpl_style(
    *,
    font_family: str = "Arial",
    font_size: int = 12,
    unicode_minus: bool = False,
    use_offset: bool = False,
) -> None:
    """Set global Matplotlib style parameters."""
    mpl.rcParams["font.family"] = font_family
    mpl.rcParams["font.size"] = font_size
    mpl.rcParams["axes.titlesize"] = font_size
    mpl.rcParams["axes.labelsize"] = font_size
    mpl.rcParams["xtick.labelsize"] = font_size
    mpl.rcParams["ytick.labelsize"] = font_size
    mpl.rcParams["legend.fontsize"] = font_size
    mpl.rcParams["axes.unicode_minus"] = unicode_minus
    mpl.rcParams["axes.formatter.useoffset"] = use_offset


def make_palette(n: int, base_colors: Optional[List[tuple]] = None) -> List[tuple]:
    """Generate n colors using a linear segmented colormap."""
    if base_colors is None:
        base_colors = BASE_COLORS
    cmap = LinearSegmentedColormap.from_list("custom", base_colors, N=max(n, 2))
    return [cmap(i) for i in range(n)]


def ion_to_tex(name: str) -> str:
    """Format ion labels like 'Mg2+' -> 'Mg$^{2+}$'."""
    match = re.match(r"^([A-Z][a-z]?)(\d*)([+-])$", name.strip())
    if match:
        sym, n, sign = match.groups()
        charge = f"{n}{sign}" if n else sign
        return rf"{sym}$^{{{charge}}}$"
    return name


def apply_corner_sci(
    ax,
    *,
    axis: str = "y",
    sigfigs: int = 3,
    fixed_exp: Optional[int] = None,
    font_size: Optional[int] = None,
) -> None:
    """Draw scientific notation multiplier in the plot corner."""
    if fixed_exp is None:
        lo, hi = ax.get_ylim() if axis == "y" else ax.get_xlim()
        maxabs = max(abs(lo), abs(hi))
        exp = 0 if (maxabs == 0 or np.isclose(maxabs, 0.0)) else int(np.floor(np.log10(maxabs)))
    else:
        exp = int(fixed_exp)

    scale = 10.0 ** exp

    def _fmt(val, _pos):
        v = val / scale
        s = f"{v:.{sigfigs}g}"
        try:
            s = f"{float(s):.{sigfigs}g}"
        except Exception:
            pass
        return s

    formatter = FuncFormatter(_fmt)
    if axis == "y":
        ax.yaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        if exp != 0:
            ax.text(
                0.0,
                1.02,
                rf"$\times 10^{{{exp}}}$",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=font_size,
            )
    else:
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        if exp != 0:
            ax.text(
                1.0,
                -0.10,
                rf"$\times 10^{{{exp}}}$",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=font_size,
            )


def plot_wavelet_components(
    series: np.ndarray,
    *,
    wavelet: str = "db3",
    level: int = 3,
    mode: str = "symmetric",
    show: bool = True,
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> None:
    """Plot original series with reconstructed wavelet components."""
    parts = decompose_wavelet_reconstructed(
        series, wavelet=wavelet, level=level, mode=mode
    )
    signals = [np.asarray(series)] + [parts[k] for k in parts]
    titles = ["Original"] + list(parts.keys())

    fig, axs = plt.subplots(len(signals), 1, figsize=(10, 3 * len(signals)))
    for ax, sig, title in zip(axs, signals, titles):
        ax.plot(sig)
        ax.set_title(title)
        ax.grid(True)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_cyclic_multi_comparison(
    signals_list: List[Dict[str, np.ndarray]],
    labels: List[str],
    *,
    num_cycles: int = 9,
    unify_x: bool = False,
    colors: Optional[List[tuple]] = None,
    font_size: int = 12,
    show: bool = True,
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> None:
    """Plot cyclic comparisons for multiple series across wavelet scales."""
    if len(signals_list) != len(labels):
        raise ValueError("signals_list and labels length mismatch")

    scales = list(signals_list[0].keys())
    n_scales = len(scales)
    n_series = len(signals_list)

    if colors is None:
        colors = make_palette(num_cycles)

    fig, axs = plt.subplots(
        n_scales,
        n_series,
        figsize=(4 * n_series, 3 * n_scales),
        squeeze=False,
        sharex=False,
        sharey=False,
    )

    for row, scale in enumerate(scales):
        for col, (signals, label) in enumerate(zip(signals_list, labels)):
            sig = np.asarray(signals[scale])
            ax = axs[row][col]

            period = max(1, len(sig) // num_cycles)
            for cycle in range(num_cycles):
                start, end = cycle * period, (cycle + 1) * period
                segment = sig[start:end]
                if len(segment) == 0:
                    continue
                segment_closed = np.append(segment, segment[0])
                x = np.arange(len(segment_closed))
                ax.plot(
                    x,
                    segment_closed,
                    color=colors[cycle % len(colors)],
                    label=f"Cycle {cycle + 1}" if (row == 0 and col == 0) else None,
                )

            if row == 0:
                ax.set_title(label)
            if col == 0:
                ax.set_ylabel(scale)

            ax.grid(False)
            apply_corner_sci(ax, axis="y", sigfigs=3, font_size=font_size)
            if unify_x:
                apply_corner_sci(ax, axis="x", sigfigs=3, font_size=font_size)

            if row != n_scales - 1:
                ax.tick_params(labelbottom=False)
            else:
                ax.tick_params(labelbottom=True)
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    handles, legend_labels = axs[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=min(num_cycles, 9),
            fontsize=font_size,
        )

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_cycle_max_min(
    scales: Iterable[str],
    cycle_max_all: Dict[str, Dict[str, List[float]]],
    cycle_min_all: Dict[str, Dict[str, List[float]]],
    *,
    num_cycles: int,
    colors: Optional[List[tuple]] = None,
    fixed_exp_by_row: Optional[Dict[int, int]] = None,
    font_size: int = 12,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Plot per-cycle maxima/minima curves for each scale."""
    scales = list(scales)
    num_samples = len(next(iter(cycle_max_all.values())))

    if colors is None:
        colors = make_palette(num_samples)

    fig, axes = plt.subplots(len(scales), 2, figsize=(16, 4 * len(scales)), sharex=False)

    for idx, scale in enumerate(scales):
        df_max = pd.DataFrame(cycle_max_all[scale], index=[f"{i + 1}" for i in range(num_cycles)])
        df_min = pd.DataFrame(cycle_min_all[scale], index=[f"{i + 1}" for i in range(num_cycles)])

        forced_exp = fixed_exp_by_row.get(idx) if fixed_exp_by_row else None

        for j, col in enumerate(df_max.columns):
            label = col if idx == 0 else None
            axes[idx, 0].plot(df_max.index, df_max[col], marker="o", color=colors[j], label=label)
        axes[idx, 0].set_title(f"{scale} - Per-cycle Max")
        axes[idx, 0].grid(False)
        apply_corner_sci(axes[idx, 0], axis="y", sigfigs=3, fixed_exp=forced_exp, font_size=font_size)

        for j, col in enumerate(df_min.columns):
            label = col if idx == 0 else None
            axes[idx, 1].plot(df_min.index, df_min[col], marker="o", color=colors[j], label=label)
        axes[idx, 1].set_title(f"{scale} - Per-cycle Min")
        axes[idx, 1].grid(False)
        apply_corner_sci(axes[idx, 1], axis="y", sigfigs=3, fixed_exp=forced_exp, font_size=font_size)

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=num_samples)

    for ax in axes.flat:
        plt.setp(ax.get_xticklabels(), rotation=0, ha="right")

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_diff_boxplots(
    scales: Iterable[str],
    df_combined: pd.DataFrame,
    cycle_max_all: Dict[str, Dict[str, List[float]]],
    cycle_min_all: Dict[str, Dict[str, List[float]]],
    *,
    fixed_exp_by_row: Optional[Dict[int, int]] = None,
    rotate_labels: int = 45,
    font_size: int = 12,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Plot boxplots of first differences for per-cycle extrema."""
    scales = list(scales)
    num_samples = len(df_combined.columns)
    colors = make_palette(num_samples)

    diff_max, diff_min = compute_first_differences(scales, cycle_max_all, cycle_min_all)

    fig, axes = plt.subplots(len(scales), 2, figsize=(16, 4 * len(scales)), sharex=False)

    for idx, scale in enumerate(scales):
        df_diff_max = pd.DataFrame(diff_max[scale])
        df_diff_min = pd.DataFrame(diff_min[scale])
        forced_exp = fixed_exp_by_row.get(idx) if fixed_exp_by_row else None

        bp1 = axes[idx, 0].boxplot(
            [df_diff_max[col].dropna() for col in df_diff_max.columns],
            patch_artist=True,
            medianprops=dict(color="red", linewidth=2),
            showfliers=True,
        )
        for box, flier, color in zip(bp1["boxes"], bp1["fliers"], colors):
            box.set_facecolor(color)
            box.set_alpha(0.85)
            flier.set(marker="o", markerfacecolor=color, markeredgecolor="black", alpha=0.8)

        axes[idx, 0].set_title(f"{scale} - Delta of Per-cycle Maxima")
        axes[idx, 0].set_xticks(range(1, num_samples + 1))
        axes[idx, 0].set_xticklabels([ion_to_tex(c) for c in df_diff_max.columns], rotation=rotate_labels)
        axes[idx, 0].grid(True, axis="y")
        apply_corner_sci(axes[idx, 0], axis="y", sigfigs=3, fixed_exp=forced_exp, font_size=font_size)

        bp2 = axes[idx, 1].boxplot(
            [df_diff_min[col].dropna() for col in df_diff_min.columns],
            patch_artist=True,
            medianprops=dict(color="red", linewidth=2),
            showfliers=True,
        )
        for box, flier, color in zip(bp2["boxes"], bp2["fliers"], colors):
            box.set_facecolor(color)
            box.set_alpha(0.85)
            flier.set(marker="o", markerfacecolor=color, markeredgecolor="black", alpha=0.8)

        axes[idx, 1].set_title(f"{scale} - Delta of Per-cycle Minima")
        axes[idx, 1].set_xticks(range(1, num_samples + 1))
        axes[idx, 1].set_xticklabels([ion_to_tex(c) for c in df_diff_min.columns], rotation=rotate_labels)
        axes[idx, 1].grid(True, axis="y")
        apply_corner_sci(axes[idx, 1], axis="y", sigfigs=3, fixed_exp=forced_exp, font_size=font_size)

    labels_pretty = [ion_to_tex(c) for c in df_combined.columns]
    legend_handles = [mpatches.Patch(color=colors[i], label=labels_pretty[i]) for i in range(num_samples)]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=num_samples, frameon=True)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
