## Repository layout

```
open_source/
├── README.md
├── LICENSE
├── requirements.txt
├── data/
│   └── README.md
├── notebooks/
│   └── demo.ipynb
├── src/
│   └── zinc_dwt/
│       ├── __init__.py
│       ├── analysis.py
│       ├── config.py
│       ├── io.py
│       ├── plotting.py
│       └── wavelet.py
└── scripts/
    └── README.md
```

## Quick start

1. Create a virtual environment (Python 3.8+ recommended).

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Place your CSV files under `data/02` (see `data/README.md`).

4. Run the analysis.

The entire analysis pipeline and paper results are calculated in the Jupyter Notebook `demo.ipynb`.

```bash
jupyter notebook notebooks/demo.ipynb
```

## Notes

- The loader assumes each CSV has the target signal in the **third column** and skips the first 6 header rows. Adjust via `--value-col` and `--skiprows` if your file layout differs.
