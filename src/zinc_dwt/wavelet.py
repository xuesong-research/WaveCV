"""Wavelet decomposition helpers."""

from __future__ import annotations

from typing import Dict, Sequence, Union

import numpy as np
import pywt

ArrayLike = Union[np.ndarray, Sequence[float]]


def decompose_wavelet_reconstructed(
    series: ArrayLike,
    *,
    wavelet: str = "db3",
    level: int = 3,
    mode: str = "symmetric",
    trim_to_original: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Return reconstructed components: A_L and D_j.

    series approx A_L + sum_j D_j
    """
    x = np.asarray(series, dtype=float)
    n = x.shape[0]

    coeffs = pywt.wavedec(x, wavelet=wavelet, level=level, mode=mode)

    def _reconstruct(coeffs_like):
        y = pywt.waverec(coeffs_like, wavelet=wavelet, mode=mode)
        if trim_to_original:
            if len(y) > n:
                y = y[:n]
            elif len(y) < n:
                y = np.pad(y, (0, n - len(y)), mode="edge")
        return y

    parts: Dict[str, np.ndarray] = {}

    # Approximation A_L
    approx_only = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
    parts[f"Approximation A_{level}"] = _reconstruct(approx_only)

    # Detail components D_j
    # coeffs = [cA_L, cD_L, cD_{L-1}, ..., cD_1]
    for j in range(1, level + 1):
        keep_detail = [np.zeros_like(coeffs[0])]
        for k in range(1, len(coeffs)):
            keep_detail.append(coeffs[k] if k == j else np.zeros_like(coeffs[k]))
        parts[f"Detail D_{j}"] = _reconstruct(keep_detail)

    return parts
