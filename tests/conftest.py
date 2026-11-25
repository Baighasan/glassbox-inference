import asyncio
from collections.abc import Iterator
from typing import Any

import httpx
import pytest

from glassbox.server.app import create_app

_SETTINGS_ENV = ("GLASSBOX_MODEL", "GLASSBOX_DEVICE", "GLASSBOX_MAX_TOKENS")


class ASGITestClient:
    """HTTP client that shares one app lifespan across requests."""

    def __init__(self) -> None:
        self.app = create_app()
        self._loop = asyncio.new_event_loop()
        self._lifespan = self.app.router.lifespan_context(self.app)
        self._loop.run_until_complete(self._lifespan.__aenter__())

    def close(self) -> None:
        try:
            self._loop.run_until_complete(self._lifespan.__aexit__(None, None, None))
        finally:
            self._loop.close()

    def request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        async def send_request() -> httpx.Response:
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                return await client.request(method, path, **kwargs)

        return self._loop.run_until_complete(send_request())

    def get(self, path: str, **kwargs: Any) -> httpx.Response:
        return self.request("GET", path, **kwargs)

    def post(self, path: str, **kwargs: Any) -> httpx.Response:
        return self.request("POST", path, **kwargs)


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[ASGITestClient]:
    for name in _SETTINGS_ENV:
        monkeypatch.delenv(name, raising=False)
    test_client = ASGITestClient()
    try:
        yield test_client
    finally:
        test_client.close()
