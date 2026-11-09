"""Chạy bằng python run.py để mở Trọ Giá trong trình duyệt."""

import argparse
import os
import subprocess
import sys
import time
import webbrowser
from urllib.error import URLError
from urllib.request import urlopen


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy web Trọ Giá trên máy này.")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    url = f"http://127.0.0.1:{args.port}"
    command = [
        sys.executable, "-B", "-m", "streamlit", "run", os.path.join(project_root, "app", "web_app.py"),
        "--server.address=127.0.0.1", f"--server.port={args.port}", "--server.headless=true",
        "--browser.gatherUsageStats=false", "--client.toolbarMode=minimal", "--theme.base=light", "--theme.primaryColor=#116457",
        "--theme.backgroundColor=#F6F7F0", "--theme.secondaryBackgroundColor=#EAF0E8",
        "--theme.textColor=#18352D",
    ]
    process = subprocess.Popen(command, cwd=project_root)
    try:
        if not args.no_browser:
            for _ in range(80):
                if process.poll() is not None:
                    break
                try:
                    with urlopen(url + "/_stcore/health", timeout=0.5) as response:
                        if response.status == 200:
                            webbrowser.open(url)
                            break
                except (URLError, TimeoutError):
                    time.sleep(0.25)
        sys.exit(process.wait())
    except KeyboardInterrupt:
        if process.poll() is None:
            process.terminate()
            process.wait()
