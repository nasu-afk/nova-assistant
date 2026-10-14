"""Filesystem tools.

Every tool here resolves its path(s) and, when a `workspace_root` is
configured, refuses to operate outside it (`check_permission` raises
`PermissionDeniedError`). This is a tool-level safeguard in addition to
whatever the security layer (Phase 7) enforces — defense in depth, not a
replacement for it.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path, PurePath
from typing import Any, ClassVar

from app.core.enums import RiskLevel
from app.core.errors import PermissionDeniedError, ToolExecutionError, ValidationError
from app.tools.base import Tool, ToolMetadata, ToolResult

# Files larger than this are refused for `read_file` to avoid loading huge
# binaries or logs into the LLM's context by accident.
MAX_READ_BYTES = 2_000_000


# Hard cap on search results handed back to the model.
MAX_SEARCH_RESULTS = 200

# Files/folders that routinely hold credentials. The LLM (and anything that can
# prompt-inject it) must never read or overwrite these through NOVA's tools,
# even inside the workspace root.
_SENSITIVE_NAMES = frozenset(
    {
        ".env",
        ".netrc",
        ".npmrc",
        ".pypirc",
        ".git-credentials",
        "id_rsa",
        "id_ed25519",
        "id_ecdsa",
        "id_dsa",
        "authorized_keys",
        "known_hosts",
        "credentials",
        "credentials.json",
        "secrets.json",
        "nova.db",
    }
)
_SENSITIVE_DIRS = frozenset({".ssh", ".aws", ".gnupg", ".azure", ".kube", ".docker"})
_SENSITIVE_SUFFIXES = (".pem", ".key", ".pfx", ".p12", ".kdbx")


def _is_sensitive(path: Path) -> bool:
    if path.name.lower() in _SENSITIVE_NAMES or path.name.lower().startswith(".env."):
        return True
    if path.suffix.lower() in _SENSITIVE_SUFFIXES:
        return True
    return any(part.lower() in _SENSITIVE_DIRS for part in path.parts)


class _PathConfinedTool(Tool):
    """Shared helper for tools that take path(s) and must stay inside a
    workspace root (set by `build_default_registry`; `None` = unconfined,
    which only tests/embedders should do) and away from credential files.

    Confinement is enforced in `check_permission`, which the registry runs
    *before* asking the user to confirm — so nobody is prompted to approve
    something that would be refused anyway — and again in `_resolve` at
    execute time as defense in depth."""

    workspace_root: Path | None = None
    # Payload keys holding paths this tool touches.
    _path_keys: ClassVar[tuple[str, ...]] = ("path",)

    def _resolve(self, raw_path: str) -> Path:
        try:
            path = Path(raw_path).expanduser().resolve()
        except (OSError, ValueError, RuntimeError) as exc:  # e.g. NUL byte, symlink loop
            raise ValidationError(f"'{raw_path}' is not a valid path.") from exc
        if self.workspace_root is not None:
            root = self.workspace_root.resolve()
            if root not in path.parents and path != root:
                raise PermissionDeniedError(
                    f"'{raw_path}' is outside the allowed workspace ({root}).",
                    user_message="I'm only allowed to work with files inside your home folder.",
                )
        if _is_sensitive(path):
            raise PermissionDeniedError(
                f"'{raw_path}' looks like a credentials/secret file; access refused.",
                user_message="I won't touch that file because it looks like it holds secrets.",
            )
        return path

    def _refuse_protected(self, raw_path: str) -> None:
        """Refuse destructive operations on the workspace root itself, the user's
        home directory, or a filesystem root — one "yes" to a confirmation prompt
        must never be enough to wipe a whole home folder."""
        path = self._resolve(raw_path)
        protected = {Path.home().resolve(), Path(path.anchor)}
        if self.workspace_root is not None:
            protected.add(self.workspace_root.resolve())
        if path in protected or path.parent == path:
            raise PermissionDeniedError(
                f"Refusing to remove or relocate protected folder '{path}'.",
                user_message="I won't delete or move your home folder or a drive root.",
            )

    def check_permission(self, payload: dict[str, Any]) -> None:
        for key in self._path_keys:
            value = payload.get(key)
            if isinstance(value, str) and value:
                self._resolve(value)

    def _exists(self, raw_path: object) -> bool:
        """Best effort: does `raw_path` already exist? (Used to decide whether a
        write/copy/move would overwrite something.)"""
        if not isinstance(raw_path, str) or not raw_path:
            return False
        try:
            return Path(raw_path).expanduser().resolve().exists()
        except (OSError, ValueError, RuntimeError):
            return False


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

    _path_keys: ClassVar[tuple[str, ...]] = ("directory",)

    def validate_input(self, payload: dict[str, Any]) -> dict[str, Any]:
        directory = payload.get("directory")
        pattern = payload.get("pattern") or "*"
        if not directory or not isinstance(directory, str):
            raise ValidationError("search_files requires a 'directory' string")
        if not isinstance(pattern, str):
            raise ValidationError("search_files 'pattern' must be a string like '*.txt'")
        if PurePath(pattern).is_absolute() or PurePath(pattern).drive:
            raise ValidationError(
                "search_files 'pattern' must be relative (e.g. '*.txt'); "
                "put the folder in 'directory' instead."
            )
        recursive = bool(payload.get("recursive", True))
        return {"directory": directory, "pattern": pattern, "recursive": recursive}

    @staticmethod
    def _search(directory: Path, pattern: str, recursive: bool) -> tuple[list[str], bool]:
        glob_fn = directory.rglob if recursive else directory.glob
        matches: list[str] = []
        try:
            for p in glob_fn(pattern):
                if _is_sensitive(p) or not p.is_file():
                    continue
                if len(matches) >= MAX_SEARCH_RESULTS:
                    return matches, True
                matches.append(str(p))
        except (OSError, ValueError, NotImplementedError) as exc:
            raise ToolExecutionError(f"Search failed: {exc}") from exc
        return matches, False

    async def execute(self, payload: dict[str, Any]) -> ToolResult:
        directory = self._resolve(payload["directory"])
        if not directory.is_dir():
            return ToolResult(success=False, error=f"'{directory}' is not a directory.")

        # rglob over a big tree is slow blocking I/O - keep it off the event loop.
        matches, truncated = await asyncio.to_thread(
            self._search, directory, payload["pattern"], payload["recursive"]
        )
        note = f" (showing the first {MAX_SEARCH_RESULTS}; narrow the pattern)" if truncated else ""
        return ToolResult(
            success=True,
            data={"matches": matches, "count": len(matches), "truncated": truncated},
            message=f"Found {len(matches)} file(s){note}.",
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
            content = await asyncio.to_thread(path.read_text, encoding="utf-8", errors="replace")
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
        content = payload.get("content", "")
        if content is None:
            content = ""
        if not isinstance(content, str):
            raise ValidationError("create_file 'content' must be a string")
        return {"path": path, "content": content}

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

    def needs_confirmation(self, payload: dict[str, Any]) -> bool:
        return self._exists(payload.get("path"))  # overwriting existing data

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

    _path_keys: ClassVar[tuple[str, ...]] = ("source", "destination")

    def needs_confirmation(self, payload: dict[str, Any]) -> bool:
        return self._exists(payload.get("destination"))  # would overwrite

    def check_permission(self, payload: dict[str, Any]) -> None:
        super().check_permission(payload)
        self._refuse_protected(payload["source"])

    def validate_input(self, payload: dict[str, Any]) -> dict[str, Any]:
        source, destination = payload.get("source"), payload.get("destination")
        if (
            not source
            or not destination
            or not isinstance(source, str)
            or not isinstance(destination, str)
        ):
            raise ValidationError("move_file requires 'source' and 'destination' strings")
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

    _path_keys: ClassVar[tuple[str, ...]] = ("source", "destination")

    def needs_confirmation(self, payload: dict[str, Any]) -> bool:
        return self._exists(payload.get("destination"))  # would overwrite

    def validate_input(self, payload: dict[str, Any]) -> dict[str, Any]:
        source, destination = payload.get("source"), payload.get("destination")
        if (
            not source
            or not destination
            or not isinstance(source, str)
            or not isinstance(destination, str)
        ):
            raise ValidationError("copy_file requires 'source' and 'destination' strings")
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

    def check_permission(self, payload: dict[str, Any]) -> None:
        super().check_permission(payload)
        self._refuse_protected(payload["path"])

    async def execute(self, payload: dict[str, Any]) -> ToolResult:
        self._refuse_protected(payload["path"])
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
