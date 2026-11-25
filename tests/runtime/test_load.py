import torch

from glassbox.config import DEFAULT_MODEL
from glassbox.runtime.runtime import Runtime


def test_load_keeps_load_time_when_weights_are_already_placed() -> None:
    runtime = Runtime()
    module = torch.nn.Linear(4, 4)
    runtime.model = module  # type: ignore[assignment]
    runtime.model_name = DEFAULT_MODEL
    runtime.model_load_seconds = 1.25

    loaded = runtime.load(DEFAULT_MODEL)

    assert loaded is module
    assert runtime.model_load_seconds == 1.25
    assert next(module.parameters()).dtype == torch.float32


def test_load_records_time_to_move_resident_weights() -> None:
    runtime = Runtime()
    module = torch.nn.Linear(4, 4)
    runtime.model = module  # type: ignore[assignment]
    runtime.model_name = DEFAULT_MODEL
    runtime.model_load_seconds = 1.25
    runtime.dtype = torch.float16

    runtime.load(DEFAULT_MODEL)

    assert runtime.model_load_seconds is not None
    assert runtime.model_load_seconds < 1.25
    assert next(module.parameters()).dtype == torch.float16
