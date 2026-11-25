from types import SimpleNamespace

import pytest
import torch

from glassbox.config import DEFAULT_MODEL
from glassbox.runtime.runtime import Runtime


class LogitsModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool = False,
    ) -> SimpleNamespace:
        del attention_mask, use_cache
        batch, seq = input_ids.shape
        logits = self.weight.view(1, 1, -1).expand(batch, seq, -1).to(input_ids.dtype)
        return SimpleNamespace(logits=logits)


def test_set_device_selects_dtype_without_moving_weights() -> None:
    runtime = Runtime()

    runtime.set_device("cpu")
    assert runtime.device == "cpu"
    assert runtime.dtype == torch.float32

    runtime.set_device("cuda")
    assert runtime.device == "cuda"
    assert runtime.dtype == torch.float16


def test_forward_on_cpu_returns_logits_without_gpu_stats() -> None:
    runtime = Runtime()
    runtime.model = LogitsModule()  # type: ignore[assignment]
    input_ids = torch.tensor([[1, 2, 3]])
    attention_mask = torch.ones_like(input_ids)

    result = runtime.forward(input_ids, attention_mask)

    assert result.logits.shape == (1, 3, 4)
    assert result.gpu_memory_allocated_mb is None
    assert result.gpu_peak_memory_allocated_mb is None
    assert result.gpu_memory_reserved_mb is None


def test_unload_clears_the_resident_model() -> None:
    runtime = Runtime()
    runtime.model = LogitsModule()  # type: ignore[assignment]
    runtime.model_name = DEFAULT_MODEL
    runtime.model_load_seconds = 1.5

    runtime.unload()

    assert runtime.model is None
    assert runtime.model_name is None
    assert runtime.model_load_seconds is None


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_load_moves_stub_to_cuda_and_forward_reports_memory() -> None:
    runtime = Runtime()
    module = LogitsModule()
    runtime.model = module  # type: ignore[assignment]
    runtime.model_name = DEFAULT_MODEL
    runtime.set_device("cuda")

    loaded = runtime.load(DEFAULT_MODEL)

    assert loaded is module
    parameter = next(module.parameters())
    assert parameter.device.type == "cuda"
    assert parameter.dtype == torch.float16

    result = runtime.forward(torch.tensor([[1, 2]]), torch.tensor([[1, 1]]))

    assert result.logits.shape == (1, 2, 4)
    assert result.gpu_memory_allocated_mb is not None
    assert result.gpu_peak_memory_allocated_mb is not None
    assert result.gpu_memory_reserved_mb is not None
