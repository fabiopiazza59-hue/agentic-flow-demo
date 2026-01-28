"""
Demo: Evaluate a server with LLM-as-judge.
"""

from playwright.sync_api import sync_playwright

BASE_URL = "http://localhost:8000"


def main():
    print("\n" + "=" * 60)
    print("  EVALUATING SERVER WITH LLM-AS-JUDGE")
    print("=" * 60)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, slow_mo=300)
        context = browser.new_context(viewport={"width": 1280, "height": 800})
        page = context.new_page()

        try:
            # Go to dashboard
            page.goto(BASE_URL)
            page.wait_for_load_state("networkidle")
            page.wait_for_timeout(1500)
            print("[OK] Loaded dashboard")

            # Track dialogs
            dialogs_received = []

            def handle_dialog(dialog):
                msg = dialog.message
                dialogs_received.append(msg)
                print(f"\n[DIALOG] {msg[:100]}...")
                dialog.accept()

            page.on("dialog", handle_dialog)

            # Click evaluate on first server
            evaluate_buttons = page.locator("button:has-text('Evaluate')")
            if evaluate_buttons.count() > 0:
                print(f"[OK] Found {evaluate_buttons.count()} servers")

                # Click the evaluate button for market_data (or first one)
                evaluate_buttons.first.click()
                print("[...] Clicked Evaluate - confirming...")

                # Wait for confirmation dialog to be handled
                page.wait_for_timeout(1000)

                print("[...] Running LLM evaluation (this may take 10-20 seconds)...")

                # Wait for evaluation result dialog
                page.wait_for_timeout(20000)

                # Print all dialogs received
                print("\n" + "=" * 60)
                print("DIALOGS RECEIVED:")
                print("=" * 60)
                for i, msg in enumerate(dialogs_received):
                    print(f"\n[Dialog {i+1}]")
                    print(msg)
                print("=" * 60)

            else:
                print("[ERROR] No servers found to evaluate")

            # Take screenshot
            page.screenshot(path="/tmp/evaluation_result.png")
            print("\n[OK] Screenshot saved to /tmp/evaluation_result.png")

            # Keep browser open
            print("\nBrowser will stay open for 5 seconds...")
            page.wait_for_timeout(5000)

        finally:
            browser.close()


if __name__ == "__main__":
    main()
