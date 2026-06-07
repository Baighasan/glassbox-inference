import uvicorn

from glassbox import main


def test_main_starts_uvicorn(monkeypatch) -> None:
    calls: list[tuple[str, str, int]] = []

    def fake_run(app: str, *, host: str, port: int) -> None:
        calls.append((app, host, port))

    monkeypatch.setattr(uvicorn, "run", fake_run)

    main()

    assert calls == [
        ("glassbox.server.app:app", "127.0.0.1", 8000),
    ]
