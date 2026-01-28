"""
Demo script: Create a custom server and evaluate an existing one.
"""

from playwright.sync_api import sync_playwright
import time

BASE_URL = "http://localhost:8000"


def create_custom_server(page):
    """Create a custom server via the UI."""
    print("\n" + "=" * 60)
    print("STEP 1: Creating a Custom Server")
    print("=" * 60)

    # Navigate to create page
    page.goto(f"{BASE_URL}/static/create.html")
    page.wait_for_load_state("networkidle")
    print("[OK] Navigated to create page")

    # Fill in server details
    page.fill('input[name="server_id"]', "weather_service")
    page.fill('input[name="server_name"]', "Weather Service")
    page.fill('textarea[name="description"]', "A custom MCP server that provides weather information for any city. Includes current conditions and forecasts.")
    page.fill('input[name="tags"]', "weather, api, utility")
    print("[OK] Filled server details")

    # Fill in the first tool (already exists)
    page.fill('input[name="tool_0_id"]', "get_current_weather")
    page.fill('input[name="tool_0_name"]', "Get Current Weather")
    page.fill('textarea[name="tool_0_description"]', "Get the current weather conditions for a specified city. Returns temperature, humidity, and conditions.")

    # Set implementation type to HTTP
    page.select_option('select[name="tool_0_impl_type"]', "http")
    page.wait_for_timeout(300)  # Wait for UI update
    page.fill('input[name="tool_0_endpoint"]', "https://api.weather.example.com/current")
    print("[OK] Filled first tool details")

    # Add a parameter to the first tool
    page.click('button:has-text("+ Add Parameter")')
    page.wait_for_timeout(200)
    page.fill('input[name="param_0_0_name"]', "city")
    page.select_option('select[name="param_0_0_type"]', "string")
    page.fill('input[name="param_0_0_desc"]', "City name to get weather for (e.g., London, New York)")
    print("[OK] Added parameter to first tool")

    # Add a second tool
    page.click('button:has-text("+ Add Tool")')
    page.wait_for_timeout(300)
    print("[OK] Added second tool")

    # Fill in second tool
    page.fill('input[name="tool_1_id"]', "get_forecast")
    page.fill('input[name="tool_1_name"]', "Get Weather Forecast")
    page.fill('textarea[name="tool_1_description"]', "Get a multi-day weather forecast for a city. Returns daily high/low temperatures and conditions.")
    page.select_option('select[name="tool_1_impl_type"]', "http")
    page.wait_for_timeout(300)
    page.fill('input[name="tool_1_endpoint"]', "https://api.weather.example.com/forecast")
    print("[OK] Filled second tool details")

    # Add parameters to second tool
    # First click the correct "Add Parameter" button (second one)
    add_param_buttons = page.locator('button:has-text("+ Add Parameter")')
    add_param_buttons.nth(1).click()
    page.wait_for_timeout(200)
    page.fill('input[name="param_1_0_name"]', "city")
    page.select_option('select[name="param_1_0_type"]', "string")
    page.fill('input[name="param_1_0_desc"]', "City name for forecast")

    add_param_buttons.nth(1).click()
    page.wait_for_timeout(200)
    page.fill('input[name="param_1_1_name"]', "days")
    page.select_option('select[name="param_1_1_type"]', "integer")
    page.fill('input[name="param_1_1_desc"]', "Number of days to forecast (1-7)")
    # Uncheck required for days
    page.uncheck('input[name="param_1_1_required"]')
    print("[OK] Added parameters to second tool")

    # Take a screenshot before submitting
    page.screenshot(path="/tmp/create_server_form.png")
    print("[OK] Screenshot saved to /tmp/create_server_form.png")

    # Submit the form
    page.click('button[type="submit"]')
    page.wait_for_timeout(1000)
    print("[OK] Form submitted")

    # Check for success (should redirect or show message)
    if "index.html" in page.url or page.url == BASE_URL + "/":
        print("[SUCCESS] Server created! Redirected to dashboard.")
    else:
        # Check for alert
        alert_text = page.locator(".alert").text_content() if page.locator(".alert").count() > 0 else ""
        if "success" in alert_text.lower():
            print(f"[SUCCESS] {alert_text}")
        else:
            print(f"[INFO] Current URL: {page.url}")

    return "weather_service"


def evaluate_existing_server(page, server_id="market_data"):
    """Evaluate an existing server."""
    print("\n" + "=" * 60)
    print(f"STEP 2: Evaluating Server '{server_id}'")
    print("=" * 60)

    # Go to main dashboard
    page.goto(BASE_URL)
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(1000)  # Wait for servers to load
    print("[OK] Navigated to dashboard")

    # Take screenshot of dashboard
    page.screenshot(path="/tmp/dashboard.png")
    print("[OK] Dashboard screenshot saved to /tmp/dashboard.png")

    # Find the evaluate button for the server
    evaluate_buttons = page.locator(f"button:has-text('Evaluate')")

    if evaluate_buttons.count() > 0:
        print(f"[OK] Found {evaluate_buttons.count()} server(s) with Evaluate button")

        # Click the first evaluate button
        # Set up dialog handler to accept the confirmation
        page.on("dialog", lambda dialog: dialog.accept())

        evaluate_buttons.first.click()
        print("[OK] Clicked Evaluate button")

        # Wait for the evaluation to complete (it makes API calls)
        page.wait_for_timeout(3000)

        # Check for alert with results
        page.wait_for_timeout(1000)
        print("[OK] Evaluation request sent")

    else:
        print("[WARN] No Evaluate buttons found, trying API directly")

        # Use API directly
        response = page.request.post(
            f"{BASE_URL}/evaluate/{server_id}",
            headers={"Content-Type": "application/json"},
            data='{"num_synthetic_cases": 3}'
        )

        if response.ok:
            result = response.json()
            print(f"\n[EVALUATION RESULTS]")
            print(f"  Server: {result.get('server_id')}")
            print(f"  Quality Score: {result.get('avg_quality_score', 'N/A')}/5")
            print(f"  Recommendation: {result.get('recommendation', 'N/A')}")
            print(f"  Reasoning: {result.get('reasoning', 'N/A')}")
        else:
            print(f"[ERROR] Evaluation failed: {response.status}")


def preview_generated_code(page, server_id):
    """Preview the generated code for a server."""
    print("\n" + "=" * 60)
    print(f"STEP 3: Preview Generated Code for '{server_id}'")
    print("=" * 60)

    # Go to dashboard
    page.goto(BASE_URL)
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(1000)

    # Click preview on the weather_service if it exists
    preview_buttons = page.locator("button:has-text('Preview')")

    if preview_buttons.count() > 0:
        # Find the row with our server and click its preview button
        server_items = page.locator(".server-item")
        for i in range(server_items.count()):
            item = server_items.nth(i)
            if server_id in item.text_content().lower() or "weather" in item.text_content().lower():
                item.locator("button:has-text('Preview')").click()
                break
        else:
            # Just click the first preview
            preview_buttons.first.click()

        page.wait_for_timeout(1000)

        # Check if modal opened
        if page.locator("#code-modal.active").count() > 0:
            code = page.locator("#code-preview").text_content()
            print("[OK] Code preview modal opened")
            print("\n--- Generated Code Preview ---")
            print(code[:1500] + "..." if len(code) > 1500 else code)
            print("--- End Preview ---\n")

            # Screenshot
            page.screenshot(path="/tmp/code_preview.png")
            print("[OK] Code preview screenshot saved to /tmp/code_preview.png")

            # Close modal
            page.click(".modal-close")
        else:
            print("[WARN] Code modal did not open")
    else:
        print("[WARN] No Preview buttons found")


def main():
    print("\n" + "=" * 60)
    print("  MCP SERVER FACTORY - DEMO SCRIPT")
    print("=" * 60)

    with sync_playwright() as p:
        # Launch browser (headed so user can see)
        browser = p.chromium.launch(headless=False, slow_mo=300)
        context = browser.new_context(viewport={"width": 1280, "height": 800})
        page = context.new_page()

        try:
            # Step 1: Create a custom server
            server_id = create_custom_server(page)
            page.wait_for_timeout(1000)

            # Step 2: Evaluate an existing server (market_data)
            evaluate_existing_server(page, "market_data")
            page.wait_for_timeout(1000)

            # Step 3: Preview generated code for our new server
            preview_generated_code(page, server_id)

            print("\n" + "=" * 60)
            print("  DEMO COMPLETE!")
            print("=" * 60)
            print("\nScreenshots saved to:")
            print("  - /tmp/create_server_form.png")
            print("  - /tmp/dashboard.png")
            print("  - /tmp/code_preview.png")
            print("\nBrowser will stay open for 10 seconds...")

            page.wait_for_timeout(10000)

        finally:
            browser.close()


if __name__ == "__main__":
    main()
