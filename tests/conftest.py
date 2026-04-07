import asyncio
import fnmatch
import time
from typing import AsyncIterator
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

try:
    import fakeredis
except ImportError:
    fakeredis = None


pytest_plugins = ["pytest_asyncio"]


@pytest.fixture(scope="session")
def event_loop_policy():
    return asyncio.DefaultEventLoopPolicy()


@pytest_asyncio.fixture
async def fakeredis_client() -> AsyncIterator["fakeredis.FakeAsyncRedis | None"]:
    if fakeredis is None:
        pytest.skip("fakeredis not installed")
    async with fakeredis.FakeAsyncRedis() as client:
        yield client


class MockUpstashRedis:
    def __init__(self):
        self._data: dict[str, str] = {}
        self._expire: dict[str, float] = {}

    async def get(self, key: str) -> str | None:
        import time

        if key in self._expire and self._expire[key] < time.time():
            del self._data[key]
            del self._expire[key]
            return None
        return self._data.get(key)

    async def set(
        self,
        key: str,
        value: str,
        ex: int | None = None,
        px: int | None = None,
        nx: bool = False,
        xx: bool = False,
    ) -> bool:
        import time

        if nx and key in self._data:
            return False
        if xx and key not in self._data:
            return False
        self._data[key] = value
        if ex:
            self._expire[key] = time.time() + ex
        elif px:
            self._expire[key] = time.time() + (px / 1000)
        return True

    async def delete(self, *keys: str) -> int:
        count = 0
        for key in keys:
            if key in self._data:
                del self._data[key]
                count += 1
            if key in self._expire:
                del self._expire[key]
        return count

    async def keys(self, pattern: str = "*") -> list[str]:
        import fnmatch

        return fnmatch.filter(self._data.keys(), pattern)

    async def exists(self, *keys: str) -> int:
        return sum(1 for key in keys if key in self._data)

    async def expire(self, key: str, seconds: int) -> bool:
        if key not in self._data:
            return False
        import time

        self._expire[key] = time.time() + seconds
        return True

    async def ttl(self, key: str) -> int:
        import time

        if key not in self._data:
            return -2
        if key not in self._expire:
            return -1
        remaining = self._expire[key] - time.time()
        return int(max(0, remaining))

    async def incr(self, key: str) -> int:
        current = int(self._data.get(key, "0"))
        self._data[key] = str(current + 1)
        return current + 1

    async def decr(self, key: str) -> int:
        current = int(self._data.get(key, "0"))
        self._data[key] = str(current - 1)
        return current - 1

    async def lpush(self, key: str, *values: str) -> int:
        self._data.setdefault(key, "")
        items = self._data[key].split("\n") if self._data[key] else []
        self._data[key] = "\n".join(list(values) + items)
        return len(self._data[key].split("\n"))

    async def rpush(self, key: str, *values: str) -> int:
        self._data.setdefault(key, "")
        items = self._data[key].split("\n") if self._data[key] else []
        self._data[key] = "\n".join(items + list(values))
        return len(self._data[key].split("\n"))

    async def lrange(self, key: str, start: int, stop: int) -> list[str]:
        if key not in self._data:
            return []
        items = self._data[key].split("\n")
        if stop == -1:
            return items[start:]
        return items[start : stop + 1]

    async def sadd(self, key: str, *members: str) -> int:
        self._data.setdefault(key, "")
        current = set(self._data[key].split("\n")) if self._data[key] else set()
        new_members = set(members) - current
        if new_members:
            self._data[key] = "\n".join(current | new_members)
        return len(new_members)

    async def smembers(self, key: str) -> set[str]:
        if key not in self._data or not self._data[key]:
            return set()
        return set(self._data[key].split("\n"))

    async def sismember(self, key: str, member: str) -> bool:
        if key not in self._data or not self._data[key]:
            return False
        return member in self._data[key].split("\n")

    async def flushdb(self) -> bool:
        self._data.clear()
        self._expire.clear()
        return True

    def clear(self) -> None:
        self._data.clear()
        self._expire.clear()


@pytest_asyncio.fixture
async def mock_upstash() -> AsyncIterator[MockUpstashRedis]:
    mock = MockUpstashRedis()
    yield mock
    mock.clear()


@pytest_asyncio.fixture
async def async_http_client() -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=MagicMock())
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
def mock_settings():
    from unittest.mock import MagicMock

    settings = MagicMock()
    settings.database_url = "postgresql+asyncpg://test:test@localhost/test"
    settings.qdrant_url = "http://localhost:6333"
    settings.qdrant_collection = "test_collection"
    settings.openai_api_key = "sk-test"
    settings.basic_model = "openai/gpt-4o-mini"
    settings.complex_model = "openai/gpt-4o"
    settings.app_name = "test_helpdesk"
    settings.log_level = "DEBUG"
    settings.upstash_redis_rest_url = "https://mock.upstash.io"
    settings.upstash_redis_rest_token = "test_token"
    return settings
