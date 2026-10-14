from __future__ import annotations

from pathlib import Path

import pytest

from app.core.errors import PermissionDeniedError, ValidationError
from app.tools.filesystem.files import (
    CopyFileTool,
    CreateDirectoryTool,
    CreateFileTool,
    DeleteFileTool,
    MoveFileTool,
    ReadFileTool,
    SearchFilesTool,
    WriteFileTool,
)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.txt").write_text("alpha")
    (tmp_path / "docs" / "b.md").write_text("bravo")
    (tmp_path / "notes.txt").write_text("root note")
    return tmp_path


@pytest.mark.asyncio
async def test_search_files_glob_recursive(workspace: Path) -> None:
    tool = SearchFilesTool()
    payload = tool.validate_input({"directory": str(workspace), "pattern": "*.txt"})

    result = await tool.execute(payload)

    assert result.success is True
    names = {Path(p).name for p in result.data["matches"]}
    assert names == {"a.txt", "notes.txt"}


@pytest.mark.asyncio
async def test_search_files_non_directory_fails(workspace: Path) -> None:
    tool = SearchFilesTool()
    payload = tool.validate_input({"directory": str(workspace / "notes.txt")})

    result = await tool.execute(payload)

    assert result.success is False


@pytest.mark.asyncio
async def test_read_file_returns_content(workspace: Path) -> None:
    tool = ReadFileTool()
    payload = tool.validate_input({"path": str(workspace / "notes.txt")})

    result = await tool.execute(payload)

    assert result.success is True
    assert result.data["content"] == "root note"


@pytest.mark.asyncio
async def test_read_file_missing_fails(workspace: Path) -> None:
    tool = ReadFileTool()
    payload = tool.validate_input({"path": str(workspace / "missing.txt")})

    result = await tool.execute(payload)

    assert result.success is False


@pytest.mark.asyncio
async def test_read_file_too_large_is_refused(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import app.tools.filesystem.files as files_mod

    monkeypatch.setattr(files_mod, "MAX_READ_BYTES", 1)
    tool = ReadFileTool()
    payload = tool.validate_input({"path": str(workspace / "notes.txt")})

    result = await tool.execute(payload)

    assert result.success is False
    assert "too large" in (result.error or "")


@pytest.mark.asyncio
async def test_create_file_creates_new_file(workspace: Path) -> None:
    tool = CreateFileTool()
    target = workspace / "new.txt"
    payload = tool.validate_input({"path": str(target), "content": "hello"})

    result = await tool.execute(payload)

    assert result.success is True
    assert target.read_text() == "hello"


@pytest.mark.asyncio
async def test_create_file_fails_if_exists(workspace: Path) -> None:
    tool = CreateFileTool()
    payload = tool.validate_input({"path": str(workspace / "notes.txt"), "content": "x"})

    result = await tool.execute(payload)

    assert result.success is False


@pytest.mark.asyncio
async def test_write_file_overwrites(workspace: Path) -> None:
    tool = WriteFileTool()
    payload = tool.validate_input({"path": str(workspace / "notes.txt"), "content": "updated"})

    result = await tool.execute(payload)

    assert result.success is True
    assert (workspace / "notes.txt").read_text() == "updated"


def test_write_file_requires_content_string() -> None:
    with pytest.raises(ValidationError):
        WriteFileTool().validate_input({"path": "x.txt"})


@pytest.mark.asyncio
async def test_create_directory_creates_nested_dirs(workspace: Path) -> None:
    tool = CreateDirectoryTool()
    target = workspace / "a" / "b" / "c"
    payload = tool.validate_input({"path": str(target)})

    result = await tool.execute(payload)

    assert result.success is True
    assert target.is_dir()


@pytest.mark.asyncio
async def test_move_file_renames(workspace: Path) -> None:
    tool = MoveFileTool()
    payload = tool.validate_input(
        {"source": str(workspace / "notes.txt"), "destination": str(workspace / "renamed.txt")}
    )

    result = await tool.execute(payload)

    assert result.success is True
    assert not (workspace / "notes.txt").exists()
    assert (workspace / "renamed.txt").exists()


@pytest.mark.asyncio
async def test_move_file_missing_source_fails(workspace: Path) -> None:
    tool = MoveFileTool()
    payload = tool.validate_input(
        {"source": str(workspace / "ghost.txt"), "destination": str(workspace / "x.txt")}
    )

    result = await tool.execute(payload)

    assert result.success is False


@pytest.mark.asyncio
async def test_copy_file_duplicates_content(workspace: Path) -> None:
    tool = CopyFileTool()
    payload = tool.validate_input(
        {"source": str(workspace / "notes.txt"), "destination": str(workspace / "copy.txt")}
    )

    result = await tool.execute(payload)

    assert result.success is True
    assert (workspace / "notes.txt").exists()
    assert (workspace / "copy.txt").read_text() == "root note"


@pytest.mark.asyncio
async def test_copy_directory_recursively(workspace: Path) -> None:
    tool = CopyFileTool()
    payload = tool.validate_input(
        {"source": str(workspace / "docs"), "destination": str(workspace / "docs_copy")}
    )

    result = await tool.execute(payload)

    assert result.success is True
    assert (workspace / "docs_copy" / "a.txt").read_text() == "alpha"


@pytest.mark.asyncio
async def test_delete_file_removes_it(workspace: Path) -> None:
    tool = DeleteFileTool()
    payload = tool.validate_input({"path": str(workspace / "notes.txt")})

    result = await tool.execute(payload)

    assert result.success is True
    assert not (workspace / "notes.txt").exists()


@pytest.mark.asyncio
async def test_delete_directory_removes_recursively(workspace: Path) -> None:
    tool = DeleteFileTool()
    payload = tool.validate_input({"path": str(workspace / "docs")})

    result = await tool.execute(payload)

    assert result.success is True
    assert not (workspace / "docs").exists()


@pytest.mark.asyncio
async def test_delete_missing_path_fails(workspace: Path) -> None:
    tool = DeleteFileTool()
    payload = tool.validate_input({"path": str(workspace / "ghost.txt")})

    result = await tool.execute(payload)

    assert result.success is False


def test_delete_file_is_high_risk_and_requires_confirmation() -> None:
    assert DeleteFileTool.metadata.requires_confirmation is True


@pytest.mark.asyncio
async def test_workspace_confinement_blocks_traversal(
    workspace: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    outside = tmp_path_factory.mktemp("outside") / "secret.txt"
    outside.write_text("nope")

    tool = ReadFileTool()
    tool.workspace_root = workspace
    payload = tool.validate_input({"path": str(outside)})

    with pytest.raises(PermissionDeniedError):
        tool._resolve(payload["path"])
