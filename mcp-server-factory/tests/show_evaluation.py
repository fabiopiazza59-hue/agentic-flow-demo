"""
Show evaluation results in browser.
"""

from playwright.sync_api import sync_playwright
import json

BASE_URL = "http://localhost:8000"


def main():
    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS VIEWER")
    print("=" * 60)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, slow_mo=200)
        context = browser.new_context(viewport={"width": 1280, "height": 800})
        page = context.new_page()

        try:
            # First, run an evaluation via API
            print("\n[1] Running evaluation on market_data server...")
            response = page.request.post(
                f"{BASE_URL}/evaluate/market_data",
                headers={"Content-Type": "application/json"},
                data=json.dumps({"num_synthetic_cases": 3})
            )

            if response.ok:
                result = response.json()
                print(f"\n" + "=" * 50)
                print("EVALUATION RESULTS")
                print("=" * 50)
                print(f"  Server: {result['server_id']}")
                print(f"  Score: {result['avg_quality_score']:.2f}/5")
                print(f"  Accuracy: {result['tool_selection_accuracy']*100:.0f}%")
                print(f"  Recommendation: {result['recommendation']}")
                print(f"  Reasoning: {result['reasoning']}")
                print("=" * 50 + "\n")
            else:
                print(f"[ERROR] Evaluation failed: {response.status}")

            # Now navigate to evaluation dashboard
            print("[2] Opening evaluation dashboard...")
            page.goto(f"{BASE_URL}/static/evaluate.html")
            page.wait_for_load_state("networkidle")
            page.wait_for_timeout(2000)

            # Take screenshot
            page.screenshot(path="/tmp/evaluation_dashboard.png")
            print("[OK] Screenshot saved to /tmp/evaluation_dashboard.png")

            # Also show the main dashboard with servers
            print("\n[3] Showing main dashboard...")
            page.goto(BASE_URL)
            page.wait_for_load_state("networkidle")
            page.wait_for_timeout(1500)

            page.screenshot(path="/tmp/main_dashboard.png")
            print("[OK] Screenshot saved to /tmp/main_dashboard.png")

            print("\n[4] Browser staying open - explore the UI!")
            print("    Try clicking 'Evaluate' on any server.")
            page.wait_for_timeout(15000)

        finally:
            browser.close()


if __name__ == "__main__":
    main()
