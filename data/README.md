# Data Repository

This directory contains the raw electrochemical data required to reproduce the results presented in the paper.

## File Placement

Please place your raw CSV files in the `02` subdirectory:
```
data/02/*.csv
```

## Expected Format

- Each CSV has **6 header rows** to skip.
- The signal used for analysis is the **third column** (index 2).
- Files are ordered by the first integer found in the filename (e.g., `sample_1.csv`, `sample_2.csv`).

If your data differs, update the corresponding parameters in `notebooks/demo.ipynb`.
