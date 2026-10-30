"""
wake_app.py - opens the Streamlit app in a real (headless) browser, the way a visitor would.
If the app is asleep, it clicks the "get this app back up" button.

Used by .github/workflows/keep_awake.yml. Needs the APP_URL environment variable.
This is a community workaround, not an official feature, and Streamlit can change
how sleeping works at any time.
"""
import os
import re
import sys

from playwright.sync_api import sync_playwright

url = os.environ.get("APP_URL", "").strip()
if not url:
    sys.exit("APP_URL is not set. Add it as a repository variable (Settings -> Variables).")

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto(url, wait_until="domcontentloaded", timeout=90_000)
    page.wait_for_timeout(10_000)

    # A sleeping app shows a page with a button asking to wake it up.
    wake_button = page.get_by_role("button", name=re.compile("get this app back up", re.I))
    if wake_button.count() > 0:
        print("App was asleep. Clicking the wake-up button.")
        wake_button.first.click()
        page.wait_for_timeout(60_000)     # give the app time to start
    else:
        print("App looks awake (no wake-up button found).")

    page.wait_for_timeout(15_000)         # stay on the page so it counts as a real visit
    print("Page title:", page.title())
    browser.close()
