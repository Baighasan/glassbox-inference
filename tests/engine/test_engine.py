import asyncio

import pytest

from glassbox.engine.engine import Engine
from glassbox.engine.exceptions import UnsupportedModelError


def test_generate_rejects_unsupported_model() -> None:
    engine = Engine()

    with pytest.raises(UnsupportedModelError, match="other-model"):
        asyncio.run(engine.generate(model_name="other-model", text="Hello"))
