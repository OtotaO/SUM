from playwright.sync_api import sync_playwright

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto("http://localhost:5001")
        page.wait_for_selector("#spectrum-slider")
        page.fill("#input-text", "The universe is fundamentally composed of mathematics and consciousness, intertwining to create the tapestry of reality we experience every day.")
        page.click("text=Book")
        page.screenshot(path="screenshot_app_action.png")
        browser.close()

run()
