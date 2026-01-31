"""Data loading helpers."""

from __future__ import annotations

from pathlib import Path
import re
import warnings
from typing import Iterable, List, Optional

import pandas as pd


def extract_number(path: Path) -> int:
    """Extract first integer from a filename for sorting."""
    match = re.search(r"(\d+)", path.name)
    return int(match.group(1)) if match else 0


def list_csv_files(target_dir: Path) -> List[Path]:
    """Return CSV files sorted by the first integer in the filename."""
    if not target_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {target_dir}")
    files = list(target_dir.glob("*.csv"))
    return sorted(files, key=extract_number)


def load_combined_csv(
    target_dir: Path,
    *,
    skiprows: int = 6,
    value_col: int = 2,
    rename_columns: Optional[Iterable[str]] = None,
    label_format: str = "S{idx}",
) -> pd.DataFrame:
    """
    Load all CSVs in a directory and combine a single column from each file.

    Parameters
    ----------
    target_dir : Path
        Directory containing CSV files.
    skiprows : int
        Number of header rows to skip for each CSV.
    value_col : int
        Zero-based column index to extract.
    rename_columns : Optional[Iterable[str]]
        If provided, rename the combined columns using this list.
    label_format : str
        Format string used for default column names when rename_columns is None.

    Returns
    -------
    pandas.DataFrame
        Combined data, one column per CSV.
    """
    files = list_csv_files(target_dir)
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {target_dir}")

    df_combined = pd.DataFrame()
    for idx, file in enumerate(files, start=1):
        try:
            df_temp = pd.read_csv(file, skiprows=skiprows)
        except Exception as exc:
            warnings.warn(f"Failed to read {file}: {exc}")
            continue

        if value_col >= df_temp.shape[1]:
            raise IndexError(
                f"value_col={value_col} out of range for file {file} with {df_temp.shape[1]} columns"
            )

        series = df_temp.iloc[:, value_col].reset_index(drop=True)
        df_combined[label_format.format(idx=idx)] = series

    if rename_columns is not None:
        rename_columns = list(rename_columns)
        if len(rename_columns) != df_combined.shape[1]:
            raise ValueError(
                "rename_columns length does not match number of loaded CSV files"
            )
        df_combined.columns = rename_columns

    return df_combined
