"""Database-backed session service for Google ADK."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import json

from sqlalchemy import Column, String, DateTime, Text, PrimaryKeyConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession

from .models import Base
from .database import async_session_factory, engine


@dataclass
class Session:
    app_name: str
    user_id: str
    session_id: str
    state: dict = field(default_factory=dict)
    create_time: Optional[datetime] = None
    update_time: Optional[datetime] = None


class DatabaseSessionService:
    def __init__(self):
        self._tables_created = False

    async def _ensure_tables(self):
        if not self._tables_created:
            async with engine.begin() as conn:
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS sessions (
                        app_name VARCHAR(100) NOT NULL,
                        user_id VARCHAR(100) NOT NULL,
                        session_id VARCHAR(100) NOT NULL,
                        state JSONB DEFAULT '{}',
                        create_time TIMESTAMP DEFAULT NOW(),
                        update_time TIMESTAMP DEFAULT NOW(),
                        PRIMARY KEY (app_name, user_id, session_id)
                    )
                    """
                )
            self._tables_created = True

    async def create_session(
        self,
        app_name: str,
        user_id: str,
        session_id: str,
        state: dict = None,
    ) -> Session:
        await self._ensure_tables()

        state = state or {}
        now = datetime.utcnow()

        async with async_session_factory() as session:
            await session.execute(
                """
                INSERT INTO sessions (app_name, user_id, session_id, state, create_time, update_time)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (app_name, user_id, session_id) DO UPDATE
                SET state = EXCLUDED.state, update_time = EXCLUDED.update_time
                """,
                [app_name, user_id, session_id, json.dumps(state), now, now],
            )
            await session.commit()

        return Session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            state=state,
            create_time=now,
            update_time=now,
        )

    async def get_session(
        self,
        app_name: str,
        user_id: str,
        session_id: str,
    ) -> Optional[Session]:
        await self._ensure_tables()

        async with async_session_factory() as session:
            result = await session.execute(
                """
                SELECT app_name, user_id, session_id, state, create_time, update_time
                FROM sessions
                WHERE app_name = $1 AND user_id = $2 AND session_id = $3
                """,
                [app_name, user_id, session_id],
            )
            row = result.fetchone()

        if row is None:
            return None

        return Session(
            app_name=row[0],
            user_id=row[1],
            session_id=row[2],
            state=row[3] if isinstance(row[3], dict) else json.loads(row[3]),
            create_time=row[4],
            update_time=row[5],
        )

    async def update_session_state(
        self,
        app_name: str,
        user_id: str,
        session_id: str,
        new_state: dict,
    ) -> Optional[Session]:
        await self._ensure_tables()

        now = datetime.utcnow()

        async with async_session_factory() as session:
            result = await session.execute(
                """
                UPDATE sessions
                SET state = $1, update_time = $2
                WHERE app_name = $3 AND user_id = $4 AND session_id = $5
                """,
                [json.dumps(new_state), now, app_name, user_id, session_id],
            )
            await session.commit()

            if result.rowcount == 0:
                return None

        return Session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            state=new_state,
            create_time=None,
            update_time=now,
        )

    async def list_sessions(
        self,
        app_name: str,
        user_id: str,
    ) -> list[Session]:
        await self._ensure_tables()

        async with async_session_factory() as session:
            result = await session.execute(
                """
                SELECT app_name, user_id, session_id, state, create_time, update_time
                FROM sessions
                WHERE app_name = $1 AND user_id = $2
                ORDER BY update_time DESC
                """,
                [app_name, user_id],
            )
            rows = result.fetchall()

        return [
            Session(
                app_name=row[0],
                user_id=row[1],
                session_id=row[2],
                state=row[3] if isinstance(row[3], dict) else json.loads(row[3]),
                create_time=row[4],
                update_time=row[5],
            )
            for row in rows
        ]

    async def delete_session(
        self,
        app_name: str,
        user_id: str,
        session_id: str,
    ) -> bool:
        await self._ensure_tables()

        async with async_session_factory() as session:
            result = await session.execute(
                """
                DELETE FROM sessions
                WHERE app_name = $1 AND user_id = $2 AND session_id = $3
                """,
                [app_name, user_id, session_id],
            )
            await session.commit()

        return result.rowcount > 0
