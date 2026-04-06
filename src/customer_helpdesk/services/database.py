from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import NullPool
from typing import AsyncGenerator
from tenacity import retry, stop_after_attempt, wait_exponential
import structlog

from .models import Base, User, Ticket, Interaction, TicketStatus
from ..config import get_settings

logger = structlog.get_logger()

settings = get_settings()

# Use NullPool for NeonDB transaction mode compatibility
engine = create_async_engine(
    settings.database_url,
    poolclass=NullPool,  # NeonDB transaction mode requirement
    echo=False,
)

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# Query functions
async def get_user_tickets(user_id: str, limit: int = 10) -> list[Ticket]:
    async with async_session_factory() as session:
        from sqlalchemy import select

        result = await session.execute(
            select(Ticket)
            .where(Ticket.user_id == user_id)
            .order_by(Ticket.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())


async def search_similar_tickets(
    query: str, category: str, limit: int = 5, days_back: int = 90
) -> list[Ticket]:
    async with async_session_factory() as session:
        from sqlalchemy import select, and_
        from datetime import datetime, timedelta

        cutoff = datetime.utcnow() - timedelta(days=days_back)

        result = await session.execute(
            select(Ticket)
            .where(
                and_(
                    Ticket.category == category,
                    Ticket.status == TicketStatus.RESOLVED,
                    Ticket.created_at >= cutoff,
                    Ticket.resolution.isnot(None),
                )
            )
            .order_by(Ticket.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
async def get_user_tickets_with_retry(user_id: str, limit: int = 10) -> list[Ticket]:
    """Wrapper with retry logic for database calls."""
    logger.info("fetching_user_tickets", user_id=user_id, limit=limit)
    return await get_user_tickets(user_id, limit)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
async def search_similar_tickets_with_retry(
    query: str, category: str, limit: int = 5, days_back: int = 90
) -> list[Ticket]:
    """Wrapper with retry logic for database calls."""
    logger.info(
        "searching_similar_tickets",
        query=query,
        category=category,
        limit=limit,
        days_back=days_back,
    )
    return await search_similar_tickets(query, category, limit, days_back)
