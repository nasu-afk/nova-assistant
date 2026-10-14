"""Async database engine + session factory.

A single `async_sessionmaker` is created per database URL. Swapping SQLite
for another SQLAlchemy-supported database later means changing
`NOVA_DATABASE_URL` — nothing above this module should know it's SQLite.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.database.models import Base


def create_engine(database_url: str) -> AsyncEngine:
    return create_async_engine(database_url, future=True)


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


async def init_models(engine: AsyncEngine) -> None:
    """Create all tables that don't exist yet. Fine for SQLite + this
    project's scale; swap for a real migration tool if the schema grows."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


class Database:
    """Convenience bundle of an engine + session factory + lifecycle."""

    def __init__(self, database_url: str) -> None:
        self.engine = create_engine(database_url)
        self.session_factory = create_session_factory(self.engine)

    async def init(self) -> None:
        await init_models(self.engine)

    async def dispose(self) -> None:
        await self.engine.dispose()

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self.session_factory() as session:
            yield session
