import asyncio
from typing import Any

import httpx
import pytest

from glassbox.config import Settings
from glassbox.server.app import create_app


class ASGITestClient:
    def __init__(self, settings: Settings) -> None:
        self.app = create_app(settings=settings)

    def request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        async def send_request() -> httpx.Response:
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                return await client.request(method, path, **kwargs)

        return asyncio.run(send_request())

    def get(self, path: str, **kwargs: Any) -> httpx.Response:
        return self.request("GET", path, **kwargs)

    def post(self, path: str, **kwargs: Any) -> httpx.Response:
        return self.request("POST", path, **kwargs)


@pytest.fixture
def client() -> ASGITestClient:
    return ASGITestClient(
        Settings(
            model="test-model",
            device="cpu",
            max_tokens=64,
        )
    )
