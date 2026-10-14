"""Memory subsystem (section 10).

- `ShortTermMemory` is a rolling in-process view for components that need a
  lightweight buffer. Durable chat turns are handled separately by
  `app.memory.conversation.ConversationHistory`.
- `LongTermMemory` persists explicit facts/preferences the user told NOVA,
  keyed for lookup, update, and deletion.
- `EpisodicMemory` persists a record of completed tasks (what was asked,
  what happened) for later recall ("what did I ask you to do yesterday?").
- `MemoryManager` is the facade the rest of the app depends on, and the
  single place that supports "inspect and delete stored memories."

A vector store can be introduced later purely inside `LongTermMemory`
(e.g. swap the lookup-by-key for a similarity search) without changing this
public interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import delete, select

from app.core.errors import MemoryError_
from app.database.engine import Database
from app.database.models import MemoryRecord, TaskRecord


@dataclass
class ShortTermMemory:
    """A small rolling window of recent conversation turns, in memory only."""

    max_turns: int = 20
    _turns: list[tuple[str, str]] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        self._turns.append((role, content))
        if len(self._turns) > self.max_turns:
            self._turns = self._turns[-self.max_turns :]

    def recent(self) -> list[tuple[str, str]]:
        return list(self._turns)

    def clear(self) -> None:
        self._turns.clear()


class LongTermMemory:
    """Persisted key/value facts ("user prefers dark mode")."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def remember(self, key: str, value: str, category: str = "general") -> str:
        record = MemoryRecord(id=uuid4().hex[:12], key=key, value=value, category=category)
        try:
            async with self._db.session() as session:
                session.add(record)
                await session.commit()
        except Exception as exc:  # noqa: BLE001
            raise MemoryError_(f"Failed to store memory '{key}': {exc}") from exc
        return record.id

    async def recall(self, key: str) -> list[MemoryRecord]:
        async with self._db.session() as session:
            result = await session.execute(select(MemoryRecord).where(MemoryRecord.key == key))
            return list(result.scalars().all())

    async def list_all(self, category: str | None = None) -> list[MemoryRecord]:
        async with self._db.session() as session:
            stmt = select(MemoryRecord).order_by(MemoryRecord.created_at, MemoryRecord.id)
            if category:
                stmt = stmt.where(MemoryRecord.category == category)
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def forget(self, memory_id: str) -> bool:
        async with self._db.session() as session:
            stmt = delete(MemoryRecord).where(MemoryRecord.id == memory_id)
            result = await session.execute(stmt)
            await session.commit()
            return bool(getattr(result, "rowcount", 0))

    async def forget_all(self) -> int:
        async with self._db.session() as session:
            result = await session.execute(delete(MemoryRecord))
            await session.commit()
            return int(getattr(result, "rowcount", 0) or 0)


class EpisodicMemory:
    """Persisted record of completed tasks, for later recall."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def record_task(self, goal: str, status: str, summary: str = "") -> str:
        task = TaskRecord(
            id=uuid4().hex[:12],
            goal=goal,
            status=status,
            summary=summary,
            completed_at=datetime.now(UTC),
        )
        async with self._db.session() as session:
            session.add(task)
            await session.commit()
        return task.id

    async def recent_tasks(self, limit: int = 20) -> list[TaskRecord]:
        async with self._db.session() as session:
            stmt = select(TaskRecord).order_by(TaskRecord.created_at.desc()).limit(limit)
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def forget_all(self) -> int:
        async with self._db.session() as session:
            result = await session.execute(delete(TaskRecord))
            await session.commit()
            return int(getattr(result, "rowcount", 0) or 0)


class MemoryManager:
    """Facade over short-term/long-term/episodic memory."""

    def __init__(self, db: Database, short_term: ShortTermMemory | None = None) -> None:
        self.short_term = short_term or ShortTermMemory()
        self.long_term = LongTermMemory(db)
        self.episodic = EpisodicMemory(db)

    async def inspect_all(self) -> dict[str, object]:
        """Return everything remembered, for a user-facing 'what do you know
        about me' view."""
        return {
            "recent_conversation": self.short_term.recent(),
            "long_term_facts": await self.long_term.list_all(),
            "recent_tasks": await self.episodic.recent_tasks(),
        }

    async def forget_everything(self) -> None:
        """Erase all persisted memory. The short-term buffer is also cleared."""
        self.short_term.clear()
        await self.long_term.forget_all()
        await self.episodic.forget_all()
