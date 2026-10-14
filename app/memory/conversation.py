"""Persistent conversation history backed by the shared NOVA database."""

from __future__ import annotations

import asyncio
import secrets
import threading
import time
from datetime import UTC, datetime
from typing import Literal, cast
from uuid import uuid4

from sqlalchemy import select, update

from app.ai.base import ChatMessage
from app.database.engine import Database
from app.database.models import Conversation, Message

_message_id_lock = threading.Lock()
_last_message_ns = 0


def _message_id() -> str:
    """A fixed-width, time-sortable id with enough randomness for uniqueness."""
    global _last_message_ns
    with _message_id_lock:
        timestamp = max(time.time_ns(), _last_message_ns + 1)
        _last_message_ns = timestamp
    return f"{timestamp:020d}{secrets.token_hex(6)}"


class ConversationHistory:
    """Owns the active conversation and persists its user/assistant turns.

    The most recent conversation is resumed on startup. Call ``start_new``
    when a UI eventually exposes an explicit "new conversation" action.
    """

    def __init__(self, db: Database) -> None:
        self._db = db
        self._conversation_id: str | None = None
        self._lock = asyncio.Lock()

    @property
    def conversation_id(self) -> str | None:
        return self._conversation_id

    async def initialize(self) -> str:
        if self._conversation_id is not None:
            return self._conversation_id
        async with self._lock:
            if self._conversation_id is not None:
                return self._conversation_id
            async with self._db.session() as session:
                result = await session.execute(
                    select(Conversation).order_by(Conversation.created_at.desc()).limit(1)
                )
                conversation = result.scalar_one_or_none()
                if conversation is None:
                    conversation = Conversation(id=uuid4().hex[:12], title="")
                    session.add(conversation)
                    await session.commit()
                self._conversation_id = conversation.id
        return self._conversation_id

    async def load_recent(self, limit: int = 20) -> list[ChatMessage]:
        conversation_id = await self.initialize()
        async with self._db.session() as session:
            result = await session.execute(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.created_at.desc(), Message.id.desc())
                .limit(limit)
            )
            records = list(reversed(result.scalars().all()))

        messages: list[ChatMessage] = []
        for record in records:
            if record.role not in {"user", "assistant", "tool"}:
                continue
            role = cast(Literal["user", "assistant", "tool"], record.role)
            messages.append(ChatMessage(role=role, content=record.content))
        return messages

    async def append(self, role: str, content: str) -> str:
        if role not in {"user", "assistant", "tool"}:
            raise ValueError(f"Unsupported persisted conversation role: {role}")
        conversation_id = await self.initialize()
        message = Message(
            id=_message_id(),
            conversation_id=conversation_id,
            role=role,
            content=content,
            created_at=datetime.now(UTC),
        )
        async with self._db.session() as session:
            session.add(message)
            if role == "user":
                title = " ".join(content.split())[:200]
                await session.execute(
                    update(Conversation)
                    .where(
                        Conversation.id == conversation_id,
                        Conversation.title == "",
                    )
                    .values(title=title)
                )
            await session.commit()
        return message.id

    async def start_new(self, title: str = "") -> str:
        conversation = Conversation(id=uuid4().hex[:12], title=title[:200])
        async with self._lock:
            async with self._db.session() as session:
                session.add(conversation)
                await session.commit()
            self._conversation_id = conversation.id
        return conversation.id

