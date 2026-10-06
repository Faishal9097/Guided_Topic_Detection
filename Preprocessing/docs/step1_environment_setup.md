# Step 1: Environment Setup

**Date:** 17-09-2026
**Author:** Adrija Pal

## What was done
- Created project folder structure on D: drive (`D:\guided-topic-detection`):
  - `data/raw/weibo_ced`, `data/raw/fakenewsnet`, `data/processed`
  - `notebooks/`, `scripts/`, `docs/`
- Created Python virtual environment (`venv`) inside the project root
- Fixed PowerShell execution policy restriction:
  - Ran `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
    to allow venv activation scripts to run
- Installed required libraries via `requirements.txt`:
  - pandas, numpy, networkx, scikit-learn, requests, gdown, tqdm,
    jupyter, ipykernel, python-dateutil
- Verified setup using `scripts/verify_setup.py` — all libraries imported
  successfully with no errors

## Verification output
Python version: 3.10.11 (tags/v3.10.11:7d4cc5a, Apr 5 2023, 00:38:17)
Python executable: D:\guided-topic-detection\venv\Scripts\python.exe
✅ pandas — version 2.3.3
✅ numpy — version 2.2.6
✅ networkx — version 3.4.2
✅ sklearn — version 1.7.2
✅ requests — version 2.34.2
✅ gdown — version 6.3.0
✅ tqdm — version 4.70.1

All required libraries are installed correctly.
## Notes / Known issues
- No GPU-specific packages installed (CPU-only setup) — not required for
  the data preparation stage.
- Jupyter kernel registration (`ipykernel install`) hung indefinitely on
  first attempt (likely antivirus/disk I/O interference after the large
  pip install). Skipped for now since it's not required until notebooks
  are used in Step 2 onward; will retry separately if needed.