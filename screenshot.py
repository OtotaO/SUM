from playwright.sync_api import sync_playwright

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto("http://localhost:5001")
        page.wait_for_selector("#spectrum-slider")
        page.screenshot(path="screenshot_app.png")
        browser.close()

run()
