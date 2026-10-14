"""Filesystem tools.

Every tool here resolves its path(s) and, when a `workspace_root` is
configured, refuses to operate outside it (`check_permission` raises
`PermissionDeniedError`). This is a tool-level safeguard in addition to
whatever the security layer (Phase 7) enforces — defense in depth, not a
replacement for it.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, ClassVar

from app.core.enums import RiskLevel
from app.core.errors import PermissionDeniedError, ToolExecutionError, ValidationError
from app.tools.base import Tool, ToolMetadata, ToolResult

# Files larger than this are refused for `read_file` to avoid loading huge
# binaries or logs into the LLM's context by accident.
MAX_READ_BYTES = 2_000_000


class _PathConfinedTool(Tool):
    """Shared helper for tools that take a `path` and must stay inside an
    optional workspace root."""

    workspace_root: Path | None = None

    def _resolve(self, raw_path: str) -> Path:
        path = Path(raw_path).expanduser().resolve()
        if self.workspace_root is not None:
            root = self.workspace_root.resolve()
            if root not in path.parents and path != root:
                raise PermissionDeniedError(
                    f"'{raw_path}' is outside the allowed workspace ({root})."
                )
        return path


class SearchFilesTool(_PathConfinedTool):
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        name="search_files",
        description="Search for files by glob pattern under a directory.",
        risk_level=RiskLevel.LOW,
        input_schema={
            "type": "object",
            "properties": {
                "directory": {"type": "string", "description": "Directory to search under"},
                "pattern": {"type": "string", "description": "Glob pattern, e.g. '*.txt'"},
                "recursive": {"type": "boolean", "description": "Search subdirectories"},
            },
            "required": ["directory"],
        },
    )

    def validate_input(self, payload: dict[str, Any]) -> dict[str, Any]:
        directory = payload.get("directory")
        pattern = payload.get("pattern", "*")
        if not directory or not isinstance(directory, str):
            raise ValidationError("search_files requires a 'directory' string")
        recursive = bool(payload.get("recursive", True))
        return {"directory": directory, "pattern": pattern, "recursive": recursive}

    async def execute(self, payload: dict[str, Any]) -> ToolResult:
        directory = self._resolve(payload["directory"])
        if not directory.is_dir():
            return ToolResult(success=False, error=f"'{directory}' is not a directory.")

        glob_fn = directory.rglob if payload["recursive"] else directory.glob
        matches = [str(p) for p in glob_fn(payload["pattern"]) if p.is_file()]
        return ToolResult(
            success=True,
            data={"matches": matches, "count": len(matches)},
            message=f"Found {len(matches)} file(s).",
        )


class ReadFileTool(_PathConfinedTool):
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        name="read_file",
        description="Read the text contents of a file.",
        risk_level=RiskLevel.LOW,
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Path to the file to read"}},
            "required": ["path"],
        },
    )

    def validate_input(self, payload: dict[str, Any]) -> dict[str, Any]:
        path = payload.get("path")
        if not path or not isinstance(path, str):
            raise ValidationError("read_file requires a 'path' string")
        return {"path": path}

    async def execute(self, payload: dict[str, Any]) -> ToolResult:
        path = self._resolve(payload["path"])
        if not path.is_file():
            return ToolResult(success=False, error=f"'{path}' is not a file.")
        size = path.stat().st_size
        if size > MAX_READ_BYTES:
            return ToolResult(success=False, error=f"'{path}' is too large to read ({size} bytes).")

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise ToolExecutionError(f"Failed to read '{path}': {exc}") from exc

        data = {"content": content, "path": str(path)}
        return ToolResult(success=True, data=data, message=f"Read {path.name}.")


class CreateFileTool(_PathConfinedTool):
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        name="create_file",
        description="Create a new file with the given text content. Fails if it already exists.",
        risk_level=RiskLevel.MEDIUM,
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path of the new file"},
                "content": {"type": "string", "description": "Text content to write"},
            },
            "required": ["path"],
        },
    )

    def validate_input(self, payload: dict[str, Any]) -> dict[str, Any]:
        path = payload.get("path")
        if not path or not isinstance(path, str):
            raise ValidationError("create_file requires a 'path' string")
        return {"path": path, "content": payload.get("content", "")}

    async def execute(self, payload: dict[str, Any]) -> ToolResult:
        path = self._resolve(payload["path"])
        if path.exists():
            return ToolResult(success=False, error=f"'{path}' already exists.")

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payload["content"], encoding="utf-8")
        except OSError as exc:
            raise ToolExecutionError(f"Failed to create '{path}': {exc}") from exc

        return ToolResult(success=True, data={"path": str(path)}, message=f"Created {path.name}.")


class WriteFileTool(_PathConfinedTool):
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        name="write_file",
        description="Overwrite (or create) a file with the given text content.",
        risk_level=RiskLevel.MEDIUM,
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path of the file to write"},
                "content": {"type": "string", "description": "Text content to write"},
            },
            "required": ["path", "content"],
        },
    )

    def validate_input(self, payload: dict[str, Any]) -> dict[str, Any]:
        path = payload.get("path")
        content = payload.get("content")
        if not path or not isinstance(path, str):
            raise ValidationError("write_file requires a 'path' string")
        if content is None or not isinstance(content, str):
            raise ValidationError("write_file requires a 'content' string")
        return {"path": path, "content": content}

    async def execute(self, payload: dict[str, Any]) -> ToolResult:
        path = self._resolve(payload["path"])
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payload["content"], encoding="utf-8")
        except OSError as exc:
            raise ToolExecutionError(f"Failed to write '{path}': {exc}") from exc

        return ToolResult(success=True, data={"path": str(path)}, message=f"Wrote {path.name}.")


class CreateDirectoryTool(_PathConfinedTool):
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        name="create_directory",
        description="Create a directory, including any missing parent directories.",
        risk_level=RiskLevel.MEDIUM,
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Directory path to create"}},
            "required": ["path"],
        },
    )

    def validate_input(self, payload: dict[str, Any]) -> dict[str, Any]:
        path = payload.get("path")
        if not path or not isinstance(path, str):
            raise ValidationError("create_directory requires a 'path' string")
        return {"path": path}

    async def execute(self, payload: dict[str, Any]) -> ToolResult:
        path = self._resolve(payload["path"])
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ToolExecutionError(f"Failed to create directory '{path}': {exc}") from exc

        return ToolResult(
            success=True, data={"path": str(path)}, message=f"Created directory {path}."
        )


class MoveFileTool(_PathConfinedTool):
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        name="move_file",
        description="Move or rename a file or directory.",
        risk_level=RiskLevel.MEDIUM,
        input_schema={
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Existing path"},
                "destination": {"type": "string", "description": "New path"},
            },
            "required": ["source", "destination"],
        },
    )

    def validate_input(self, payload: dict[str, Any]) -> dict[str, Any]:
        source, destination = payload.get("source"), payload.get("destination")
        if not source or not destination:
            raise ValidationError("move_file requires 'source' and 'destination'")
        return {"source": source, "destination": destination}

    async def execute(self, payload: dict[str, Any]) -> ToolResult:
        source = self._resolve(payload["source"])
        destination = self._resolve(payload["destination"])
        if not source.exists():
            return ToolResult(success=False, error=f"'{source}' does not exist.")

        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
        except OSError as exc:
            raise ToolExecutionError(
                f"Failed to move '{source}' to '{destination}': {exc}"
            ) from exc

        return ToolResult(success=True, message=f"Moved {source.name} to {destination}.")


class CopyFileTool(_PathConfinedTool):
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        name="copy_file",
        description="Copy a file or directory to a new location.",
        risk_level=RiskLevel.MEDIUM,
        input_schema={
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Existing path"},
                "destination": {"type": "string", "description": "Destination path"},
            },
            "required": ["source", "destination"],
        },
    )

    def validate_input(self, payload: dict[str, Any]) -> dict[str, Any]:
        source, destination = payload.get("source"), payload.get("destination")
        if not source or not destination:
            raise ValidationError("copy_file requires 'source' and 'destination'")
        return {"source": source, "destination": destination}

    async def execute(self, payload: dict[str, Any]) -> ToolResult:
        source = self._resolve(payload["source"])
        destination = self._resolve(payload["destination"])
        if not source.exists():
            return ToolResult(success=False, error=f"'{source}' does not exist.")

        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(source, destination, dirs_exist_ok=True)
            else:
                shutil.copy2(source, destination)
        except OSError as exc:
            raise ToolExecutionError(
                f"Failed to copy '{source}' to '{destination}': {exc}"
            ) from exc

        return ToolResult(success=True, message=f"Copied {source.name} to {destination}.")


class DeleteFileTool(_PathConfinedTool):
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        name="delete_file",
        description="Permanently delete a file or directory. Irreversible.",
        risk_level=RiskLevel.HIGH,
        requires_confirmation=True,
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Path to delete"}},
            "required": ["path"],
        },
    )

    def validate_input(self, payload: dict[str, Any]) -> dict[str, Any]:
        path = payload.get("path")
        if not path or not isinstance(path, str):
            raise ValidationError("delete_file requires a 'path' string")
        return {"path": path}

    async def execute(self, payload: dict[str, Any]) -> ToolResult:
        path = self._resolve(payload["path"])
        if not path.exists():
            return ToolResult(success=False, error=f"'{path}' does not exist.")

        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        except OSError as exc:
            raise ToolExecutionError(f"Failed to delete '{path}': {exc}") from exc

        return ToolResult(success=True, message=f"Deleted {path}.")
