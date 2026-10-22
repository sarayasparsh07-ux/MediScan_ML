"""Visual QA: capture every dashboard page in headless Chromium.

Usage: python tests/screenshot_pages.py <url> <out_dir> [width]
(Start the app first: streamlit run app.py --server.port 8599)
"""
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

PAGES = ["Start Here", "Dataset & EDA", "Model Leaderboard", "Confusion Matrix Lab",
         "PCA & Decision Boundaries", "Scaling & Feature Insights", "Significance & Thresholds",
         "Model Combinations", "Clinical Recommendation", "Methodology"]

url, out = sys.argv[1], Path(sys.argv[2])
width = int(sys.argv[3]) if len(sys.argv) > 3 else 1440
out.mkdir(parents=True, exist_ok=True)
chrome = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"

with sync_playwright() as p:
    b = p.chromium.launch(executable_path=chrome if Path(chrome).exists() else None)
    pg = b.new_page(viewport={"width": width, "height": 900})
    pg.goto(url, wait_until="networkidle", timeout=240_000)
    pg.wait_for_selector("text=Diagnostic Intelligence Lab", timeout=240_000)
    pg.wait_for_selector(".js-plotly-plot", timeout=240_000)
    for i, name in enumerate(PAGES):
        try:
            pg.locator("section[data-testid='stSidebar']").get_by_text(name, exact=False).first.click(timeout=5000)
        except Exception:
            btn = pg.locator("[data-testid='stExpandSidebarButton'], [data-testid='collapsedControl']").first
            btn.click()
            time.sleep(0.6)
            pg.locator("section[data-testid='stSidebar']").get_by_text(name, exact=False).first.click()
            time.sleep(0.4)
            pg.keyboard.press("Escape")
        if width < 700:  # close the mobile sidebar drawer so the page content is visible
            for sel in ["[data-testid='stSidebarCollapseButton'] button", "[data-testid='stSidebarCollapseButton']"]:
                try:
                    pg.locator(sel).first.click(timeout=2000)
                    break
                except Exception:
                    continue
        time.sleep(2.5)
        pg.wait_for_load_state("networkidle")
        h = pg.evaluate("""() => { const m = document.querySelector('[data-testid="stMain"]') ||
                          document.querySelector('section.main'); return m ? m.scrollHeight : document.body.scrollHeight; }""")
        pg.set_viewport_size({"width": width, "height": min(max(900, h + 60), 16000)})
        time.sleep(2.0)
        pg.screenshot(path=str(out / f"{i + 1:02d}_{name.replace(' ', '_').replace('&', 'and')}.png"), full_page=True)
        pg.set_viewport_size({"width": width, "height": 900})
        print("captured", name, h)
    b.close()
