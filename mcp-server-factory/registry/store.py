"""
File-based registry store with YAML storage.

Provides CRUD operations for servers, tools, and evaluations.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional
import uuid

import yaml

from .schemas import ServerDefinition, EvaluationResult


class ServerStore:
    """File-based server registry with YAML storage."""

    def __init__(self, base_path: Path):
        """
        Initialize the server store.

        Args:
            base_path: Base path for registry data storage
        """
        self.base_path = Path(base_path)
        self.servers_path = self.base_path / "servers"
        self.archive_path = self.base_path / "archive"
        self.servers_path.mkdir(parents=True, exist_ok=True)

    def get(self, server_id: str) -> Optional[ServerDefinition]:
        """
        Get server by ID.

        Args:
            server_id: Server identifier

        Returns:
            ServerDefinition or None if not found
        """
        path = self.servers_path / f"{server_id}.yaml"
        if not path.exists():
            return None
        data = yaml.safe_load(path.read_text())
        return ServerDefinition(**data)

    def save(self, server: ServerDefinition) -> None:
        """
        Save server definition.

        Args:
            server: ServerDefinition to save
        """
        server.updated_at = datetime.utcnow()
        path = self.servers_path / f"{server.id}.yaml"

        # Convert to dict with JSON-serializable datetimes
        data = server.model_dump(mode="json")

        # Write YAML with nice formatting
        path.write_text(
            yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True)
        )

    def list_all(self, tags: Optional[list[str]] = None) -> list[ServerDefinition]:
        """
        List all servers, optionally filtered by tags.

        Args:
            tags: Optional list of tags to filter by (OR logic)

        Returns:
            List of ServerDefinition objects
        """
        servers = []
        for path in self.servers_path.glob("*.yaml"):
            # Skip templates (files starting with _)
            if path.name.startswith("_"):
                continue
            try:
                data = yaml.safe_load(path.read_text())
                server = ServerDefinition(**data)
                if tags is None or any(t in server.tags for t in tags):
                    servers.append(server)
            except Exception as e:
                print(f"[ServerStore] Error loading {path}: {e}")
                continue

        # Sort by updated_at descending
        servers.sort(key=lambda s: s.updated_at, reverse=True)
        return servers

    def delete(self, server_id: str) -> bool:
        """
        Delete server (moves to archive).

        Args:
            server_id: Server identifier

        Returns:
            True if deleted, False if not found
        """
        path = self.servers_path / f"{server_id}.yaml"
        if path.exists():
            self.archive_path.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            archive_path = self.archive_path / f"{server_id}_{timestamp}.yaml"
            path.rename(archive_path)
            return True
        return False

    def exists(self, server_id: str) -> bool:
        """Check if server exists."""
        path = self.servers_path / f"{server_id}.yaml"
        return path.exists()


class EvaluationStore:
    """File-based evaluation results storage."""

    def __init__(self, base_path: Path):
        """
        Initialize the evaluation store.

        Args:
            base_path: Base path for registry data storage
        """
        self.base_path = Path(base_path)
        self.evaluations_path = self.base_path / "evaluations"
        self.evaluations_path.mkdir(parents=True, exist_ok=True)

    def save(self, evaluation: EvaluationResult) -> str:
        """
        Save evaluation result.

        Args:
            evaluation: EvaluationResult to save

        Returns:
            Evaluation ID
        """
        if not evaluation.id:
            evaluation.id = str(uuid.uuid4())[:8]

        # Store by server_id/evaluation_id.yaml
        server_dir = self.evaluations_path / evaluation.server_id
        server_dir.mkdir(parents=True, exist_ok=True)

        path = server_dir / f"{evaluation.id}.yaml"
        data = evaluation.model_dump(mode="json")
        path.write_text(
            yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True)
        )
        return evaluation.id

    def get(self, server_id: str, evaluation_id: str) -> Optional[EvaluationResult]:
        """Get evaluation by ID."""
        path = self.evaluations_path / server_id / f"{evaluation_id}.yaml"
        if not path.exists():
            return None
        data = yaml.safe_load(path.read_text())
        return EvaluationResult(**data)

    def list_for_server(
        self, server_id: str, limit: int = 10
    ) -> list[EvaluationResult]:
        """List evaluations for a server, most recent first."""
        server_dir = self.evaluations_path / server_id
        if not server_dir.exists():
            return []

        evaluations = []
        for path in server_dir.glob("*.yaml"):
            try:
                data = yaml.safe_load(path.read_text())
                evaluations.append(EvaluationResult(**data))
            except Exception as e:
                print(f"[EvaluationStore] Error loading {path}: {e}")
                continue

        # Sort by timestamp descending
        evaluations.sort(key=lambda e: e.timestamp, reverse=True)
        return evaluations[:limit]

    def list_all(self, limit: int = 50) -> list[EvaluationResult]:
        """List all evaluations across all servers."""
        evaluations = []
        for server_dir in self.evaluations_path.iterdir():
            if server_dir.is_dir():
                evaluations.extend(self.list_for_server(server_dir.name, limit=limit))

        evaluations.sort(key=lambda e: e.timestamp, reverse=True)
        return evaluations[:limit]
