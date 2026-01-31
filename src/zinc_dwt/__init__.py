"""Zinc DWT analysis package."""

from .config import DEFAULT_LABELS, DEFAULT_LEVEL, DEFAULT_NUM_CYCLES, DEFAULT_WAVELET
from .io import load_combined_csv, list_csv_files
from .analysis import (
    compute_autocorrelation,
    compute_cycle_extrema,
    compute_cycle_extrema_for_df,
    compute_first_differences,
    compute_monotonicity,
)
from .wavelet import decompose_wavelet_reconstructed
from .plotting import (
    apply_corner_sci,
    ion_to_tex,
    make_palette,
    plot_cyclic_multi_comparison,
    plot_cycle_max_min,
    plot_diff_boxplots,
    plot_wavelet_components,
    set_mpl_style,
)

__all__ = [
    "DEFAULT_LABELS",
    "DEFAULT_LEVEL",
    "DEFAULT_NUM_CYCLES",
    "DEFAULT_WAVELET",
    "load_combined_csv",
    "list_csv_files",
    "compute_autocorrelation",
    "compute_cycle_extrema",
    "compute_cycle_extrema_for_df",
    "compute_first_differences",
    "compute_monotonicity",
    "decompose_wavelet_reconstructed",
    "apply_corner_sci",
    "ion_to_tex",
    "make_palette",
    "plot_cyclic_multi_comparison",
    "plot_cycle_max_min",
    "plot_diff_boxplots",
    "plot_wavelet_components",
    "set_mpl_style",
]
