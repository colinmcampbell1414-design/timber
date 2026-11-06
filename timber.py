"""TIMBER desktop launcher: opens the app in its own window (no browser, no command line).

How it works:
1. This program starts the app's engine (Streamlit) as a hidden background
   process on a free port of this computer (it is never visible on the network:
   it only listens on 127.0.0.1, which means "this computer only").
2. It opens a desktop window (pywebview, which uses the Edge WebView2 that comes
   with Windows 10/11) showing "Starting TIMBER..." until the engine is ready,
   then shows the app inside the window.
3. Closing the window stops the engine.

Run it without building:   .\\.venv\\Scripts\\python timber.py
Build TIMBER.exe:           .\\build_exe.ps1   (see README "Desktop app")
"""

import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

APP_TITLE = "TIMBER - Tree Instability Mapping By Elevation & Reflectance"
FROZEN = getattr(sys, "frozen", False)  # True when running as the built TIMBER.exe

# Where things are:
# - HOME: the folder holding TIMBER.exe (or this file).
# - BUNDLE: where the build put app.py, timber_ui/ and the theme (inside TIMBER's _internal folder).
HOME = Path(sys.executable).parent if FROZEN else Path(__file__).parent
BUNDLE = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))


def user_folders():
    r"""Where surveys, saved results and the log go. Three ways TIMBER can be started:

    - from the project folder (python timber.py): data/ and .timber_cache/ there;
    - a PORTABLE TIMBER folder (it has a file named portable.txt, e.g. dist\TIMBER
      copied to a USB stick): everything next to TIMBER.exe;
    - INSTALLED (e.g. in Program Files, which programs can't write to): surveys in
      Documents\TIMBER\data, saved results and the log in %LOCALAPPDATA%\TIMBER.
    Returns (data folder, cache folder, log file).
    """
    if not FROZEN:
        return HOME / "data", HOME / ".timber_cache", HOME / "timber.log"
    if (HOME / "portable.txt").exists():
        return HOME / "data", HOME / "cache", HOME / "timber.log"
    documents = Path(os.environ.get("USERPROFILE", str(Path.home()))) / "Documents"
    local = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))) / "TIMBER"
    return documents / "TIMBER" / "data", local / "cache", local / "timber.log"


DATA_DIR, CACHE_DIR, LOG_FILE = user_folders()

LOADING_PAGE = """
<html><body style="background:#0e1411;color:#e6eee8;font-family:Segoe UI,sans-serif;
display:flex;align-items:center;justify-content:center;height:100vh;margin:0">
<div style="text-align:center"><div style="font-size:64px">&#127794;</div>
<h1 style="margin:8px 0">TIMBER</h1>
<p>Tree Instability Mapping By Elevation &amp; Reflectance</p>
<p style="color:#9aa0a6">Starting... (the first start can take up to a minute)</p></div>
</body></html>
"""


def free_port() -> int:
    """Ask Windows for a port number nothing else is using."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_server(port: int):
    """Background engine: run the Streamlit app on `port` (called with --server)."""
    from streamlit.web import cli as stcli

    sys.argv = [
        "streamlit", "run", str(BUNDLE / "app.py"),
        "--server.port", str(port),
        "--server.address", "127.0.0.1",  # this computer only
        "--server.headless", "true",  # don't open a browser; the window shows the app
        "--server.fileWatcherType", "none",
        "--global.developmentMode", "false",
        "--browser.gatherUsageStats", "false",
        "--client.toolbarMode", "minimal",  # hide Streamlit's developer menu
    ]  # colors and fonts come from .streamlit/config.toml
    sys.exit(stcli.main())


def start_server(port: int) -> subprocess.Popen:
    """Start the engine as a hidden background process, writing its messages to timber.log."""
    command = [sys.executable, "--server", str(port)] if FROZEN else [sys.executable, __file__, "--server", str(port)]
    env = dict(os.environ, TIMBER_DATA_DIR=str(DATA_DIR), TIMBER_CACHE_DIR=str(CACHE_DIR))
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    log = open(LOG_FILE, "w", encoding="utf-8", errors="replace")
    # cwd = BUNDLE so Streamlit finds TIMBER's theme in .streamlit/config.toml.
    return subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=str(BUNDLE),
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def wait_until_ready(port: int, server: subprocess.Popen, timeout_s: float = 180.0) -> bool:
    """Check the engine's health page twice a second until it answers "ok"."""
    url = f"http://127.0.0.1:{port}/_stcore/health"
    start = time.time()
    while time.time() - start < timeout_s:
        if server.poll() is not None:  # the engine stopped: something went wrong
            return False
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.read().strip() == b"ok":
                    return True
        except OSError:
            pass
        time.sleep(0.5)
    return False


def main():
    import webview

    DATA_DIR.mkdir(parents=True, exist_ok=True)  # where survey folders go
    port = free_port()
    server = start_server(port)
    webview.settings["ALLOW_DOWNLOADS"] = True  # so "Download CSV / GeoTIFF / PNG" buttons work
    window = webview.create_window(APP_TITLE, html=LOADING_PAGE, width=1400, height=900, min_size=(900, 600))

    def load_app():
        if wait_until_ready(port, server):
            window.load_url(f"http://127.0.0.1:{port}")
        else:
            window.load_html(LOADING_PAGE.replace(
                "Starting... (the first start can take up to a minute)",
                f"TIMBER could not start. Details are in {LOG_FILE}"))

    try:
        webview.start(load_app)  # runs until the window is closed
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--server":
        run_server(int(sys.argv[2]))
    else:
        main()
