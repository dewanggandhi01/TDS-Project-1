---
title: "Web Automation with Playwright"
original_url: "https://tds.s-anand.net/#/web-automation-with-playwright?id=example-scraping-a-jsrendered-site"
downloaded_at: "2025-06-13T17:06:09.355185"
---
[Web Scraping with Playwright in Python](#/web-automation-with-playwright?id=web-scraping-with-playwright-in-python)
--------------------------------------------------------------------------------------------------------------------

Scrape JavaScript‑heavy sites effortlessly with Playwright.

[![🤖 Playwright: Advanced Web Scraping in Python (14 min)](https://i.ytimg.com/vi_webp/biFzRHk4xpY/sddefault.webp)](https://youtu.be/biFzRHk4xpY) ([youtube.com](https://www.youtube.com/watch?v=biFzRHk4xpY&utm_source=chatgpt.com))

Playwright offers:

* **JavaScript rendering**: Executes page scripts so you scrape only after content appears. ([playwright.dev](https://playwright.dev/python/docs/intro))
* **Headless & headed modes**: Run without UI or in a real browser for debugging. ([playwright.dev](https://playwright.dev/python/docs/intro))
* **Auto‑waiting & retry**: Built‑in locators reduce flakiness. ([playwright.dev](https://playwright.dev/python/docs/locators))
* **Multi‑browser support**: Chromium, Firefox, WebKit—all from one API. ([playwright.dev](https://playwright.dev/python/docs/intro))

### [Example: Scraping a JS‑Rendered Site](#/web-automation-with-playwright?id=example-scraping-a-jsrendered-site)

We’ll scrape [Quotes to Scrape (JS)](https://quotes.toscrape.com/js/)—a site that loads quotes via JavaScript, so a simple `requests` call gets only an empty shell ([quotes.toscrape.com](https://quotes.toscrape.com/js/)). Playwright runs the scripts and gives us the real content:

```
# /// script
# dependencies = ["playwright"]
# ///

from playwright.sync_api import sync_playwright

def scrape_quotes():
    with sync_playwright() as p:
        # Channel can be "chrome", "msedge", "chrome-beta", "msedge-beta" or "msedge-dev".
        browser = p.chromium.launch(headless=True, channel="chrome")
        page = browser.new_page()
        page.goto("https://quotes.toscrape.com/js/")
        quotes = page.query_selector_all(".quote")
        for q in quotes:
            text = q.query_selector(".text").inner_text()
            author = q.query_selector(".author").inner_text()
            print(f"{text} — {author}")
        browser.close()

if __name__ == "__main__":
    scrape_quotes()Copy to clipboardErrorCopied
```

Save as `scraper.py` and run:

```
uv run scraper.pyCopy to clipboardErrorCopied
```

You’ll see each quote plus author printed—fetched only after the JS executes.

[Previous

LLM Video Screen-Scraping](#/llm-video-screen-scraping)

[Next

Scheduled Scraping with GitHub Actions](#/scheduled-scraping-with-github-actions)