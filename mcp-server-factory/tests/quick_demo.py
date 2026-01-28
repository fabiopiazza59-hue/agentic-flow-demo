"""
Quick demo showing dashboards.
"""

from playwright.sync_api import sync_playwright

BASE_URL = "http://localhost:8000"


def main():
    print("\n" + "=" * 60)
    print("  MCP SERVER FACTORY - QUICK DEMO")
    print("=" * 60)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, slow_mo=200)
        context = browser.new_context(viewport={"width": 1280, "height": 800})
        page = context.new_page()

        try:
            # Show evaluation dashboard
            print("\n[1] Evaluation Dashboard")
            page.goto(f"{BASE_URL}/static/evaluate.html")
            page.wait_for_load_state("networkidle")
            page.wait_for_timeout(2000)
            page.screenshot(path="/tmp/eval_dashboard.png")
            print("    Screenshot: /tmp/eval_dashboard.png")

            # Show main dashboard
            print("\n[2] Main Dashboard with Servers")
            page.goto(BASE_URL)
            page.wait_for_load_state("networkidle")
            page.wait_for_timeout(2000)
            page.screenshot(path="/tmp/main_dashboard.png")
            print("    Screenshot: /tmp/main_dashboard.png")

            # Preview weather_service code
            print("\n[3] Previewing Weather Service Code...")
            server_items = page.locator(".server-item")
            for i in range(server_items.count()):
                text = server_items.nth(i).text_content()
                if "weather" in text.lower():
                    server_items.nth(i).locator("button:has-text('Preview')").click()
                    break

            page.wait_for_timeout(1500)
            if page.locator("#code-modal.active").count() > 0:
                page.screenshot(path="/tmp/code_preview.png")
                print("    Screenshot: /tmp/code_preview.png")
                page.keyboard.press("Escape")

            # Show templates
            print("\n[4] Templates")
            page.click("button:has-text('Browse Templates')")
            page.wait_for_timeout(1500)
            page.screenshot(path="/tmp/templates.png")
            print("    Screenshot: /tmp/templates.png")

            print("\n" + "=" * 60)
            print("  DEMO COMPLETE - Browser staying open")
            print("=" * 60)
            page.wait_for_timeout(10000)

        finally:
            browser.close()


if __name__ == "__main__":
    main()
