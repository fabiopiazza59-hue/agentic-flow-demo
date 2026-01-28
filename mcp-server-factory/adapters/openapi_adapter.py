"""
OpenAPI to MCP Server Adapter.

Convert OpenAPI specifications to MCP server definitions.
"""

import re
from typing import Optional

import httpx

from registry.schemas import (
    ServerDefinition,
    ToolDefinition,
    ToolParameter,
    ToolImplementation,
)


class OpenAPIAdapter:
    """Convert OpenAPI specs to MCP server definitions."""

    TYPE_MAPPING = {
        "string": "string",
        "integer": "integer",
        "number": "number",
        "boolean": "boolean",
        "array": "array",
        "object": "object",
    }

    async def import_from_url(
        self,
        spec_url: str,
        server_id: str,
        server_name: str,
        include_patterns: Optional[list[str]] = None,
        exclude_patterns: Optional[list[str]] = None,
    ) -> ServerDefinition:
        """
        Import OpenAPI spec and create server definition.

        Args:
            spec_url: URL to OpenAPI spec (JSON format)
            server_id: ID for the new server
            server_name: Name for the new server
            include_patterns: Path patterns to include (e.g., ["/users", "/orders"])
            exclude_patterns: Path patterns to exclude (e.g., ["/admin", "/internal"])

        Returns:
            ServerDefinition with tools for each endpoint
        """
        # Fetch spec
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(spec_url)
            response.raise_for_status()
            spec = response.json()

        return self.import_from_spec(
            spec=spec,
            server_id=server_id,
            server_name=server_name,
            spec_url=spec_url,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
        )

    def import_from_spec(
        self,
        spec: dict,
        server_id: str,
        server_name: str,
        spec_url: Optional[str] = None,
        include_patterns: Optional[list[str]] = None,
        exclude_patterns: Optional[list[str]] = None,
    ) -> ServerDefinition:
        """
        Create server definition from parsed OpenAPI spec.

        Args:
            spec: Parsed OpenAPI spec dict
            server_id: ID for the new server
            server_name: Name for the new server
            spec_url: Original URL (for reference)
            include_patterns: Path patterns to include
            exclude_patterns: Path patterns to exclude

        Returns:
            ServerDefinition with tools
        """
        # Get base URL
        base_url = ""
        if "servers" in spec and spec["servers"]:
            base_url = spec["servers"][0].get("url", "")

        # Get description
        info = spec.get("info", {})
        description = info.get("description", f"MCP Server imported from OpenAPI spec")
        if spec_url:
            description += f"\n\nSource: {spec_url}"

        # Extract tools from paths
        tools = []
        for path, methods in spec.get("paths", {}).items():
            for method, operation in methods.items():
                if method not in ["get", "post", "put", "patch", "delete"]:
                    continue

                # Apply filters
                if include_patterns and not any(p in path for p in include_patterns):
                    continue
                if exclude_patterns and any(p in path for p in exclude_patterns):
                    continue

                tool = self._operation_to_tool(path, method, operation, base_url, spec)
                if tool:
                    tools.append(tool)

        return ServerDefinition(
            id=server_id,
            name=server_name,
            version=info.get("version", "1.0.0"),
            description=description,
            source_type="openapi",
            tools=tools,
            tags=["imported", "openapi"],
        )

    def _operation_to_tool(
        self,
        path: str,
        method: str,
        operation: dict,
        base_url: str,
        spec: dict,
    ) -> Optional[ToolDefinition]:
        """Convert OpenAPI operation to tool definition."""
        # Get or generate operation ID
        operation_id = operation.get("operationId")
        if not operation_id:
            # Generate from path and method
            clean_path = path.replace("/", "_").replace("{", "").replace("}", "")
            operation_id = f"{method}{clean_path}"

        # Clean up operation ID to be valid Python identifier
        tool_id = self._to_snake_case(operation_id)

        # Build description
        summary = operation.get("summary", "")
        description = operation.get("description", "")
        if summary and description:
            full_description = f"{summary}\n\n{description}"
        else:
            full_description = summary or description or f"Call {method.upper()} {path}"

        # Ensure minimum description length
        if len(full_description) < 10:
            full_description = f"Execute {method.upper()} request to {path}"

        # Extract parameters
        params = self._extract_parameters(operation, spec)

        # Build endpoint URL
        endpoint = f"{base_url}{path}"

        return ToolDefinition(
            id=tool_id,
            name=operation.get("operationId", tool_id),
            description=full_description,
            parameters=params,
            implementation=ToolImplementation(
                type="http",
                endpoint=endpoint,
                method=method.upper(),
            ),
        )

    def _extract_parameters(self, operation: dict, spec: dict) -> list[ToolParameter]:
        """Extract parameters from operation."""
        params = []

        # Query/Path parameters
        for param in operation.get("parameters", []):
            # Handle $ref
            if "$ref" in param:
                param = self._resolve_ref(param["$ref"], spec)
                if not param:
                    continue

            param_schema = param.get("schema", {})
            param_type = self._map_type(param_schema.get("type", "string"))

            params.append(
                ToolParameter(
                    name=param["name"],
                    type=param_type,
                    description=param.get("description", f"Parameter: {param['name']}"),
                    required=param.get("required", param.get("in") == "path"),
                    default=param_schema.get("default"),
                )
            )

        # Request body parameters
        request_body = operation.get("requestBody", {})
        if request_body:
            content = request_body.get("content", {})
            json_content = content.get("application/json", {})
            schema = json_content.get("schema", {})

            # Handle $ref
            if "$ref" in schema:
                schema = self._resolve_ref(schema["$ref"], spec) or {}

            # Extract properties from schema
            properties = schema.get("properties", {})
            required_fields = schema.get("required", [])

            for prop_name, prop_schema in properties.items():
                # Handle nested $ref
                if "$ref" in prop_schema:
                    prop_schema = self._resolve_ref(prop_schema["$ref"], spec) or {}

                params.append(
                    ToolParameter(
                        name=prop_name,
                        type=self._map_type(prop_schema.get("type", "string")),
                        description=prop_schema.get("description", f"Body parameter: {prop_name}"),
                        required=prop_name in required_fields,
                        default=prop_schema.get("default"),
                    )
                )

        return params

    def _resolve_ref(self, ref: str, spec: dict) -> Optional[dict]:
        """Resolve a JSON reference."""
        if not ref.startswith("#/"):
            return None

        parts = ref[2:].split("/")
        current = spec
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current

    def _map_type(self, openapi_type: str) -> str:
        """Map OpenAPI type to our schema type."""
        return self.TYPE_MAPPING.get(openapi_type, "string")

    def _to_snake_case(self, name: str) -> str:
        """Convert camelCase or PascalCase to snake_case."""
        # Insert underscore before uppercase letters
        s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
        # Insert underscore before uppercase letters followed by lowercase
        s2 = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1)
        # Convert to lowercase and clean up
        result = s2.lower()
        # Remove leading/trailing underscores and collapse multiple underscores
        result = re.sub(r"_+", "_", result).strip("_")
        # Ensure it starts with a letter
        if result and result[0].isdigit():
            result = "op_" + result
        return result or "unknown_operation"
