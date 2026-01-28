"""
Code generator for MCP servers.

Uses Jinja2 templates to generate Python code from server definitions.
"""

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from registry.schemas import ServerDefinition


def python_type_filter(type_str: str) -> str:
    """Convert schema type to Python type annotation."""
    mapping = {
        "string": "str",
        "number": "float",
        "integer": "int",
        "boolean": "bool",
        "object": "dict",
        "array": "list",
    }
    return mapping.get(type_str, "Any")


class ServerGenerator:
    """Generate MCP server Python code from definitions."""

    def __init__(self, templates_path: Path, output_path: Path):
        """
        Initialize the generator.

        Args:
            templates_path: Path to Jinja2 templates directory
            output_path: Path to output generated code
        """
        self.templates_path = Path(templates_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Setup Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(self.templates_path),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Add custom filters
        self.env.filters["python_type"] = python_type_filter

    def generate(self, server: ServerDefinition) -> Path:
        """
        Generate server Python file.

        Args:
            server: ServerDefinition to generate code for

        Returns:
            Path to generated file
        """
        template = self.env.get_template("server_base.py.j2")
        code = template.render(server=server)

        output_file = self.output_path / f"{server.id}_server.py"
        output_file.write_text(code)

        return output_file

    def preview(self, server: ServerDefinition) -> str:
        """
        Preview generated code without saving.

        Args:
            server: ServerDefinition to preview

        Returns:
            Generated code as string
        """
        template = self.env.get_template("server_base.py.j2")
        return template.render(server=server)

    def generate_handler_stub(self, server: ServerDefinition) -> Path:
        """
        Generate a handlers stub file with placeholder implementations.

        Args:
            server: ServerDefinition to generate handlers for

        Returns:
            Path to generated handlers file
        """
        handlers_dir = self.output_path / "handlers"
        handlers_dir.mkdir(parents=True, exist_ok=True)

        # Generate handler file
        lines = [
            '"""',
            f"Handler stubs for {server.name}",
            "",
            "Implement these functions to provide actual tool functionality.",
            '"""',
            "",
        ]

        for tool in server.tools:
            if tool.implementation.type == "python":
                # Generate async handler stub
                params = ", ".join(
                    f"{p.name}: {python_type_filter(p.type)}"
                    for p in tool.parameters
                )
                lines.extend([
                    f"async def {tool.id}_handler({params}) -> dict:",
                    f'    """',
                    f"    {tool.description}",
                    f'    """',
                    f"    # TODO: Implement",
                    f'    return {{"status": "not_implemented"}}',
                    "",
                    "",
                ])

        output_file = handlers_dir / f"{server.id}_handlers.py"
        output_file.write_text("\n".join(lines))

        return output_file

    def list_generated(self) -> list[dict[str, Any]]:
        """
        List all generated server files.

        Returns:
            List of dicts with file info
        """
        files = []
        for path in self.output_path.glob("*_server.py"):
            files.append({
                "name": path.stem,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "modified": path.stat().st_mtime,
            })
        return sorted(files, key=lambda f: f["modified"], reverse=True)
