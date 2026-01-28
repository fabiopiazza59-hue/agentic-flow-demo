"""
End-to-end tests for MCP Server Factory using Playwright.
"""

import pytest
from playwright.sync_api import Page, expect


BASE_URL = "http://localhost:8000"


class TestDashboard:
    """Test the main dashboard page."""

    def test_homepage_loads(self, page: Page):
        """Test that the homepage loads correctly."""
        page.goto(BASE_URL)

        # Check title
        expect(page.locator("h1")).to_contain_text("MCP Server Factory")

        # Check quick action cards exist
        expect(page.locator(".card")).to_have_count(3)

        # Check navigation links
        expect(page.locator("text=API Docs")).to_be_visible()

    def test_servers_list_loads(self, page: Page):
        """Test that the servers list section loads."""
        page.goto(BASE_URL)

        # Wait for servers to load
        page.wait_for_selector(".servers-list")

        # Check the servers header
        expect(page.locator(".servers-header h2")).to_contain_text("Your Servers")

        # Refresh button should exist
        expect(page.locator("button:has-text('Refresh')")).to_be_visible()


class TestTemplates:
    """Test the templates functionality."""

    def test_browse_templates_modal(self, page: Page):
        """Test opening the templates modal."""
        page.goto(BASE_URL)

        # Click browse templates button
        page.click("button:has-text('Browse Templates')")

        # Modal should appear
        expect(page.locator("#templates-modal")).to_be_visible()

        # Wait for templates to load
        page.wait_for_selector("#templates-container .card")

        # Should have templates listed
        expect(page.locator("#templates-container .card")).to_have_count(4)

    def test_create_from_template(self, page: Page):
        """Test creating a server from a template."""
        page.goto(BASE_URL)

        # Open templates modal
        page.click("button:has-text('Browse Templates')")
        page.wait_for_selector("#templates-container .card")

        # Click use template on the first one (Market Data)
        page.on("dialog", lambda dialog: dialog.accept("test_market"))

        # This will trigger a prompt - we need to handle it
        page.evaluate("""
            window.prompt = (msg, def_val) => 'test_market_' + Date.now();
        """)

        page.click("#templates-container button:has-text('Use Template')")


class TestImportOpenAPI:
    """Test the OpenAPI import functionality."""

    def test_import_modal_opens(self, page: Page):
        """Test that the import modal opens."""
        page.goto(BASE_URL)

        # Click import button
        page.click("button:has-text('Import API')")

        # Modal should appear
        expect(page.locator("#import-modal")).to_be_visible()

        # Form fields should exist
        expect(page.locator("input[name='spec_url']")).to_be_visible()
        expect(page.locator("input[name='server_id']")).to_be_visible()
        expect(page.locator("input[name='server_name']")).to_be_visible()

    def test_import_form_validation(self, page: Page):
        """Test form validation on import modal."""
        page.goto(BASE_URL)

        page.click("button:has-text('Import API')")
        page.wait_for_selector("#import-modal.active")

        # Try to submit empty form
        page.click("#import-form button[type='submit']")

        # Form should not submit (HTML5 validation)
        expect(page.locator("#import-modal")).to_be_visible()


class TestCreateServer:
    """Test the create server page."""

    def test_create_page_loads(self, page: Page):
        """Test that the create page loads."""
        page.goto(f"{BASE_URL}/static/create.html")

        # Check heading
        expect(page.locator("h1")).to_contain_text("Create MCP Server")

        # Form should exist
        expect(page.locator("#create-form")).to_be_visible()

        # Initial tool should be added
        expect(page.locator(".tool-item")).to_have_count(1)

    def test_add_tool(self, page: Page):
        """Test adding a tool."""
        page.goto(f"{BASE_URL}/static/create.html")

        # Should start with 1 tool
        expect(page.locator(".tool-item")).to_have_count(1)

        # Add another tool
        page.click("button:has-text('+ Add Tool')")

        # Should now have 2 tools
        expect(page.locator(".tool-item")).to_have_count(2)

    def test_add_parameter_to_tool(self, page: Page):
        """Test adding parameters to a tool."""
        page.goto(f"{BASE_URL}/static/create.html")

        # Add a parameter
        page.click("button:has-text('+ Add Parameter')")

        # Parameter fields should appear
        expect(page.locator(".param-item")).to_have_count(1)

        # Add another parameter
        page.click("button:has-text('+ Add Parameter')")
        expect(page.locator(".param-item")).to_have_count(2)

    def test_remove_tool(self, page: Page):
        """Test removing a tool."""
        page.goto(f"{BASE_URL}/static/create.html")

        # Add another tool first
        page.click("button:has-text('+ Add Tool')")
        expect(page.locator(".tool-item")).to_have_count(2)

        # Remove the first tool
        page.click(".tool-item:first-child button:has-text('Remove')")
        expect(page.locator(".tool-item")).to_have_count(1)


class TestAPI:
    """Test the API endpoints via the UI."""

    def test_health_endpoint(self, page: Page):
        """Test the health endpoint returns correctly."""
        response = page.request.get(f"{BASE_URL}/health")
        assert response.ok

        data = response.json()
        assert data["status"] == "healthy"
        assert "components" in data

    def test_list_servers_api(self, page: Page):
        """Test listing servers via API."""
        response = page.request.get(f"{BASE_URL}/servers")
        assert response.ok

        data = response.json()
        assert isinstance(data, list)

    def test_list_templates_api(self, page: Page):
        """Test listing templates via API."""
        response = page.request.get(f"{BASE_URL}/templates")
        assert response.ok

        data = response.json()
        assert len(data) == 4  # We defined 4 templates

    def test_create_and_delete_server(self, page: Page):
        """Test creating and deleting a server via API."""
        import time
        import json

        # Create a unique server
        server_id = f"test_server_{int(time.time())}"

        # Create server (must use JSON, not form data)
        create_response = page.request.post(
            f"{BASE_URL}/servers",
            headers={"Content-Type": "application/json"},
            data=json.dumps({
                "id": server_id,
                "name": "Test Server",
                "description": "A test server created by Playwright",
                "tags": ["test"],
                "tools": []
            })
        )
        assert create_response.status == 201

        # Get server
        get_response = page.request.get(f"{BASE_URL}/servers/{server_id}")
        assert get_response.ok

        data = get_response.json()
        assert data["id"] == server_id
        assert data["name"] == "Test Server"

        # Delete server
        delete_response = page.request.delete(f"{BASE_URL}/servers/{server_id}")
        assert delete_response.ok

        # Verify deleted
        get_deleted = page.request.get(f"{BASE_URL}/servers/{server_id}")
        assert get_deleted.status == 404

    def test_preview_server_code(self, page: Page):
        """Test previewing generated code for a server."""
        # First ensure market_data server exists
        response = page.request.get(f"{BASE_URL}/servers/market_data")

        if response.status == 404:
            pytest.skip("market_data server not found")

        # Preview code
        preview_response = page.request.post(f"{BASE_URL}/servers/market_data/preview")
        assert preview_response.ok

        data = preview_response.json()
        assert "code" in data
        assert "FastMCP" in data["code"]
        assert "market_data" in data["code"].lower() or "Market Data" in data["code"]


class TestEvaluations:
    """Test the evaluation functionality."""

    def test_evaluations_page_loads(self, page: Page):
        """Test that the evaluations page loads."""
        page.goto(f"{BASE_URL}/static/evaluate.html")

        # Check heading
        expect(page.locator("h1")).to_contain_text("Evaluation Dashboard")

        # Stats should be visible
        expect(page.locator(".stats-grid")).to_be_visible()

    def test_list_evaluations_api(self, page: Page):
        """Test listing evaluations via API."""
        response = page.request.get(f"{BASE_URL}/evaluate")
        assert response.ok

        data = response.json()
        assert "evaluations" in data
        assert "total" in data


class TestCodePreview:
    """Test the code preview functionality."""

    def test_preview_modal(self, page: Page):
        """Test the code preview modal."""
        page.goto(BASE_URL)

        # Wait for servers to load
        page.wait_for_timeout(1000)

        # Check if there are any servers with preview buttons
        preview_buttons = page.locator("button:has-text('Preview')")

        if preview_buttons.count() > 0:
            # Click the first preview button
            preview_buttons.first.click()

            # Modal should appear
            page.wait_for_selector("#code-modal.active")
            expect(page.locator("#code-modal")).to_be_visible()

            # Code should be present
            expect(page.locator("#code-preview")).not_to_be_empty()


# Pytest fixtures
@pytest.fixture(scope="session")
def browser_context_args(browser_context_args):
    """Configure browser context."""
    return {
        **browser_context_args,
        "viewport": {"width": 1280, "height": 720},
    }
