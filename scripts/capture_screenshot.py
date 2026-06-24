"""Capture a screenshot of the running Streamlit demo for the README.

Launches the app on a free port, drives it with headless Chromium (Playwright):
types a question, waits for the grounded answer + Sources panel to render, and
saves docs/screenshot.png. Reproducible — re-run any time the UI changes.

Usage: python scripts/capture_screenshot.py   (requires a built index + API key)
"""

import socket
import subprocess
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).parent.parent
OUT = ROOT / "docs" / "screenshot.png"
QUESTION = "How does Dense Passage Retrieval encode questions and passages for retrieval?"


def _free_port() -> int:
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def main() -> None:
    port = _free_port()
    proc = subprocess.Popen(
        [".venv/bin/streamlit", "run", "app/streamlit_app.py",
         "--server.port", str(port), "--server.headless", "true",
         "--browser.gatherUsageStats", "false"],
        cwd=str(ROOT),
    )
    try:
        url = f"http://localhost:{port}"
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1280, "height": 1100})
            # Wait for the server to accept connections.
            for _ in range(60):
                try:
                    page.goto(url, timeout=2000)
                    break
                except Exception:
                    time.sleep(1)
            page.get_by_role("textbox").first.fill(QUESTION)
            page.keyboard.press("Enter")
            # Wait for the answer + Sources panel (the app renders "Sources (" once done).
            sources = page.get_by_text("Sources (", exact=False)
            sources.wait_for(timeout=120_000)
            sources.click()  # expand the panel so the page/section citations show
            time.sleep(2)  # let markdown + expander settle
            OUT.parent.mkdir(exist_ok=True)
            page.screenshot(path=str(OUT), full_page=True)
            browser.close()
        print(f"wrote {OUT}")
    finally:
        proc.terminate()
        proc.wait()


if __name__ == "__main__":
    main()
