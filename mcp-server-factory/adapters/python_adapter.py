"""
Python Module to MCP Server Adapter.

Convert Python modules with typed functions to MCP server definitions.
"""

import inspect
import importlib
import re
from typing import Optional, get_type_hints, Any

from registry.schemas import (
    ServerDefinition,
    ToolDefinition,
    ToolParameter,
    ToolImplementation,
)


class PythonAdapter:
    """Convert Python modules to MCP server definitions."""

    TYPE_MAPPING = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        "str": "string",
        "int": "integer",
        "float": "number",
        "bool": "boolean",
        "list": "array",
        "dict": "object",
    }

    async def import_from_module(
        self,
        module_path: str,
        server_id: str,
        server_name: str,
        function_filter: Optional[list[str]] = None,
    ) -> ServerDefinition:
        """
        Import MCP server from a Python module.

        Args:
            module_path: Dotted module path (e.g., 'my_package.tools')
            server_id: ID for the new server
            server_name: Name for the new server
            function_filter: Only import these function names (None = all public)

        Returns:
            ServerDefinition with tools from module functions
        """
        # Import module
        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            raise ValueError(f"Failed to import module '{module_path}': {e}")

        # Get module docstring for description
        description = inspect.getdoc(module) or f"MCP Server from {module_path}"

        # Extract functions
        tools = []
        for name, obj in inspect.getmembers(module):
            # Skip private/magic functions
            if name.startswith("_"):
                continue

            # Skip non-functions
            if not (inspect.isfunction(obj) or inspect.iscoroutinefunction(obj)):
                continue

            # Apply filter
            if function_filter and name not in function_filter:
                continue

            tool = self._function_to_tool(name, obj, module_path)
            if tool:
                tools.append(tool)

        return ServerDefinition(
            id=server_id,
            name=server_name,
            description=description,
            source_type="custom",
            tools=tools,
            tags=["imported", "python"],
        )

    def _function_to_tool(
        self, name: str, func: Any, module_path: str
    ) -> Optional[ToolDefinition]:
        """Convert a Python function to a tool definition."""
        # Get docstring
        docstring = inspect.getdoc(func) or f"Function {name} from {module_path}"

        # Ensure minimum description length
        if len(docstring) < 10:
            docstring = f"Execute the {name} function. {docstring}"

        # Get parameters from signature
        try:
            sig = inspect.signature(func)
            type_hints = get_type_hints(func) if hasattr(func, "__annotations__") else {}
        except (ValueError, TypeError):
            sig = None
            type_hints = {}

        params = []
        if sig:
            for param_name, param in sig.parameters.items():
                # Skip self/cls
                if param_name in ("self", "cls"):
                    continue

                # Get type
                param_type = type_hints.get(param_name, str)
                mapped_type = self._map_type(param_type)

                # Get default
                has_default = param.default is not inspect.Parameter.empty
                default_value = param.default if has_default else None

                # Determine if required
                required = not has_default

                # Try to extract description from docstring
                param_desc = self._extract_param_description(docstring, param_name)
                if not param_desc:
                    param_desc = f"Parameter: {param_name}"

                params.append(
                    ToolParameter(
                        name=param_name,
                        type=mapped_type,
                        description=param_desc,
                        required=required,
                        default=default_value if has_default else None,
                    )
                )

        # Convert function name to valid tool ID
        tool_id = self._to_snake_case(name)

        return ToolDefinition(
            id=tool_id,
            name=name,
            description=docstring,
            parameters=params,
            implementation=ToolImplementation(
                type="python",
                handler=f"{module_path}:{name}",
            ),
        )

    def _map_type(self, python_type: Any) -> str:
        """Map Python type to schema type."""
        # Handle direct type matches
        if python_type in self.TYPE_MAPPING:
            return self.TYPE_MAPPING[python_type]

        # Handle string type names
        type_name = getattr(python_type, "__name__", str(python_type))
        if type_name in self.TYPE_MAPPING:
            return self.TYPE_MAPPING[type_name]

        # Handle Optional/Union types
        origin = getattr(python_type, "__origin__", None)
        if origin is not None:
            # Get the first non-None type from Union
            args = getattr(python_type, "__args__", ())
            for arg in args:
                if arg is not type(None):
                    return self._map_type(arg)

        # Default to string
        return "string"

    def _extract_param_description(self, docstring: str, param_name: str) -> Optional[str]:
        """Extract parameter description from docstring."""
        if not docstring:
            return None

        # Look for Google-style docstring format
        # Args:
        #     param_name: Description here
        patterns = [
            rf"{param_name}\s*:\s*(.+?)(?:\n\s*\w+:|$)",  # Google style
            rf":param\s+{param_name}\s*:\s*(.+?)(?:\n|$)",  # Sphinx style
            rf"{param_name}\s+--\s+(.+?)(?:\n|$)",  # Numpy style
        ]

        for pattern in patterns:
            match = re.search(pattern, docstring, re.IGNORECASE | re.DOTALL)
            if match:
                desc = match.group(1).strip()
                # Clean up multi-line descriptions
                desc = " ".join(desc.split())
                return desc

        return None

    def _to_snake_case(self, name: str) -> str:
        """Convert to snake_case."""
        s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
        s2 = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1)
        result = s2.lower()
        result = re.sub(r"_+", "_", result).strip("_")
        if result and result[0].isdigit():
            result = "func_" + result
        return result or "unknown_function"
