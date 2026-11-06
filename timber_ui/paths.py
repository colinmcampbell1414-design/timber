"""Where TIMBER keeps its files.

- Surveys:  the data folder (one subfolder per survey: DTM, DSM, orthophoto).
- Cache:    prepared surveys and saved analysis results, so things open fast
            the second time. Safe to delete; TIMBER just works them out again.

The launcher (timber.py) decides the folders and passes them in environment
variables, because they depend on how TIMBER was started:
- from the project folder (python timber.py / streamlit run app.py):
  data/ and .timber_cache/ in the project folder;
- portable TIMBER folder (dist\\TIMBER, e.g. on a USB stick): next to TIMBER.exe;
- installed in Program Files (future installer): Documents\\TIMBER\\data and
  %LOCALAPPDATA%\\TIMBER\\cache, because Program Files can't be written to.
"""

import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("TIMBER_DATA_DIR") or PROJECT / "data")
CACHE_DIR = Path(os.environ.get("TIMBER_CACHE_DIR") or PROJECT / ".timber_cache")
ASSETS = PROJECT / "assets"


def survey_folders() -> list[Path]:
    """Every subfolder of the data folder that holds GeoTIFF files, A to Z."""
    if not DATA_ROOT.exists():
        return []
    return sorted(p for p in DATA_ROOT.iterdir() if p.is_dir() and any(p.glob("*.tif*")))
