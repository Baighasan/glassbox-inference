from unittest.mock import Mock

import pytest
import uvicorn

from glassbox import main


@pytest.mark.parametrize(
    ("argv", "reload_dirs", "glassbox_level", "uvicorn_level"),
    [([], None, None, None), (["--reload"], ["src"], "DEBUG", "debug")],
)
def test_main_starts_uvicorn(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    reload_dirs: list[str] | None,
    glassbox_level: str | None,
    uvicorn_level: str | None,
) -> None:
    monkeypatch.delenv("GLASSBOX_LOGGING_LEVEL", raising=False)
    monkeypatch.delenv("UVICORN_LOG_LEVEL", raising=False)
    if glassbox_level is not None:
        monkeypatch.setenv("GLASSBOX_LOGGING_LEVEL", glassbox_level)
    if uvicorn_level is not None:
        monkeypatch.setenv("UVICORN_LOG_LEVEL", uvicorn_level)

    run = Mock()
    monkeypatch.setattr(uvicorn, "run", run)

    main(argv)

    run.assert_called_once()
    args, kwargs = run.call_args
    assert args == ("glassbox.server.app:app",)
    assert kwargs["host"] == "127.0.0.1"
    assert kwargs["port"] == 8000
    assert kwargs["reload"] is (reload_dirs is not None)
    assert kwargs["reload_dirs"] == reload_dirs
    assert kwargs["log_config"]["loggers"]["glassbox"]["level"] == (
        glassbox_level or "INFO"
    )
    assert kwargs["log_level"] == uvicorn_level
