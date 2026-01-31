"""Analysis helpers for cycle statistics and monotonicity."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from .wavelet import decompose_wavelet_reconstructed

def compute_autocorrelation(series: Iterable[float], lag: int | None = None) -> Tuple[float, int]:
    """Compute autocorrelation at a given lag; default lag is len(series)//9."""
    s = pd.Series(series)
    if lag is None:
        lag = max(1, len(s) // 9)
    return s.autocorr(lag=lag), lag


def compute_cycle_extrema(
    signal: np.ndarray, num_cycles: int
) -> Tuple[List[float], List[float]]:
    """Compute per-cycle max/min for a 1D signal."""
    cycle_len = max(1, len(signal) // num_cycles)
    max_list: List[float] = []
    min_list: List[float] = []
    for i in range(num_cycles):
        start = i * cycle_len
        end = (i + 1) * cycle_len if i < num_cycles - 1 else len(signal)
        segment = signal[start:end]
        if len(segment) == 0:
            continue
        max_list.append(float(np.max(segment)))
        min_list.append(float(np.min(segment)))
    return max_list, min_list


def compute_cycle_extrema_for_df(
    df: pd.DataFrame,
    *,
    wavelet: str = "db3",
    level: int = 3,
    mode: str = "symmetric",
    num_cycles: int = 9,
) -> Tuple[List[str], Dict[str, Dict[str, List[float]]], Dict[str, Dict[str, List[float]]]]:
    """
    Compute per-cycle maxima/minima for each column and each wavelet scale.

    Returns
    -------
    scales : list[str]
    cycle_max_all : dict[scale][column] -> list[float]
    cycle_min_all : dict[scale][column] -> list[float]
    """
    scales: List[str] = []
    cycle_max_all: Dict[str, Dict[str, List[float]]] = {}
    cycle_min_all: Dict[str, Dict[str, List[float]]] = {}

    for col in df.columns:
        signals = decompose_wavelet_reconstructed(
            df[col].values, wavelet=wavelet, level=level, mode=mode
        )
        if not scales:
            scales = list(signals.keys())
            cycle_max_all = {scale: {} for scale in scales}
            cycle_min_all = {scale: {} for scale in scales}

        for scale in scales:
            max_list, min_list = compute_cycle_extrema(signals[scale], num_cycles)
            cycle_max_all[scale][col] = max_list
            cycle_min_all[scale][col] = min_list

    return scales, cycle_max_all, cycle_min_all


def compute_monotonicity(
    scales: Iterable[str],
    cycle_max_all: Dict[str, Dict[str, List[float]]],
    cycle_min_all: Dict[str, Dict[str, List[float]]],
    num_cycles: int,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """Compute Kendall tau monotonicity for per-cycle max/min series."""
    monotonicity_max: Dict[str, Dict[str, float]] = {scale: {} for scale in scales}
    monotonicity_min: Dict[str, Dict[str, float]] = {scale: {} for scale in scales}

    for scale in scales:
        for col, max_list in cycle_max_all[scale].items():
            tau_max, _ = kendalltau(range(num_cycles), max_list)
            monotonicity_max[scale][col] = float(tau_max) if tau_max is not None else float("nan")
        for col, min_list in cycle_min_all[scale].items():
            tau_min, _ = kendalltau(range(num_cycles), min_list)
            monotonicity_min[scale][col] = float(tau_min) if tau_min is not None else float("nan")

    return monotonicity_max, monotonicity_min


def compute_first_differences(
    scales: Iterable[str],
    cycle_max_all: Dict[str, Dict[str, List[float]]],
    cycle_min_all: Dict[str, Dict[str, List[float]]],
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, np.ndarray]]]:
    """Compute first differences of per-cycle maxima and minima."""
    diff_max: Dict[str, Dict[str, np.ndarray]] = {scale: {} for scale in scales}
    diff_min: Dict[str, Dict[str, np.ndarray]] = {scale: {} for scale in scales}

    for scale in scales:
        for col in cycle_max_all[scale]:
            diff_max[scale][col] = np.diff(np.array(cycle_max_all[scale][col]))
            diff_min[scale][col] = np.diff(np.array(cycle_min_all[scale][col]))

    return diff_max, diff_min
