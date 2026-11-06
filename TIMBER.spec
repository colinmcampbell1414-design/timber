# PyInstaller build recipe for TIMBER.exe (run it with build_exe.ps1).
#
# The result is a folder, dist\TIMBER\, holding TIMBER.exe plus everything it
# needs (Python and all libraries), so it runs on any Windows 10/11 PC without
# installing anything. Copy the whole folder, not just the exe.
#
# Why the long lists below: PyInstaller finds what to pack by following the
# "import" lines in timber.py. The app itself (app.py) is started later by the
# engine, so its imports, the terrain and timber_ui packages, TIMBER's theme
# (.streamlit/config.toml), and the data files some libraries need (Streamlit's
# web page, rasterio's map-projection tables, Plotly's chart definitions, the
# click-on-map component) must be listed by hand.

import os
import sys

from PyInstaller.utils.hooks import collect_all, collect_submodules, copy_metadata

# Always pack THIS folder's code, whichever folder the build was started from
# (SPECPATH = the folder holding this file).
sys.path.insert(0, SPECPATH)
os.chdir(SPECPATH)

datas = [("app.py", "."), ("assets/timber.png", "assets"), (".streamlit/config.toml", ".streamlit")]
binaries = []
hiddenimports = collect_submodules("terrain") + collect_submodules("timber_ui") + [
    "streamlit.web.cli", "streamlit.runtime.scriptrunner.magic_funcs",
    "matplotlib.backends.backend_agg", "scipy.stats", "scipy.spatial", "scipy.ndimage",
    "PIL.ImageDraw", "webview",
]
for package in ["streamlit", "streamlit_image_coordinates", "rasterio", "skimage", "pyproj", "altair", "pyarrow",
                "plotly"]:
    try:
        d, b, h = collect_all(package)
    except Exception:
        continue  # a package that isn't installed (e.g. pyproj) is simply skipped
    datas += d
    binaries += b
    hiddenimports += h
for package in ["streamlit", "rasterio", "numpy", "pandas", "pyarrow", "altair", "pywebview", "plotly"]:
    try:
        datas += copy_metadata(package)  # Streamlit checks library versions at start-up
    except Exception:
        pass

a = Analysis(
    ["timber.py"],
    pathex=[SPECPATH],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["pytest", "tkinter", "IPython", "notebook", "PyQt5", "PyQt6", "PySide2", "PySide6"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="TIMBER",
    icon="assets/timber.ico",
    console=False,  # a window app: no black command-line window
)
coll = COLLECT(exe, a.binaries, a.datas, name="TIMBER")
