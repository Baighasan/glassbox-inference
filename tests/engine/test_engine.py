import asyncio
from types import SimpleNamespace

import pytest
import torch

from glassbox.config import DEFAULT_MODEL
from glassbox.engine.engine import Engine
from glassbox.engine.exceptions import UnsupportedModelError


class FakeTokenizer:
    def __init__(self, prompt_ids: list[int], eos_token_id: int | list[int] | None) -> None:
        self.prompt_ids = prompt_ids
        self.eos_token_id = eos_token_id

    def __call__(self, text: str, return_tensors: str = "pt") -> SimpleNamespace:
        del text, return_tensors
        input_ids = torch.tensor([self.prompt_ids])
        return SimpleNamespace(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return " ".join(str(int(token_id)) for token_id in token_ids.tolist())


class FakeRuntime:
    def __init__(self, next_ids: list[int]) -> None:
        self.next_ids = next_ids
        self.forward_calls = 0
        self.load_calls = 0
        self.device = "cpu"
        self.dtype = torch.float32
        self.model: object | None = None
        self.model_name: str | None = None
        self.model_load_seconds: float | None = None

    def set_device(self, device: str) -> None:
        self.device = device
        self.dtype = torch.float16 if device == "cuda" else torch.float32

    def load(self, model_name: str) -> object:
        self.load_calls += 1
        self.model_name = model_name
        self.model = object()
        self.model_load_seconds = 0.1
        return self.model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        del attention_mask
        token_id = self.next_ids[self.forward_calls]
        self.forward_calls += 1
        vocab = max(self.next_ids) + 1
        logits = torch.full((input_ids.shape[0], input_ids.shape[1], vocab), -1.0)
        logits[:, -1, token_id] = 10.0
        return SimpleNamespace(
            logits=logits,
            gpu_memory_allocated_mb=None,
            gpu_peak_memory_allocated_mb=None,
            gpu_memory_reserved_mb=None,
        )

    def unload(self) -> None:
        self.model = None
        self.model_name = None
        self.model_load_seconds = None


def _engine(runtime: FakeRuntime) -> Engine:
    engine = Engine()
    engine.runtime = runtime  # type: ignore[assignment]
    return engine


def test_generate_rejects_unsupported_model_before_load() -> None:
    runtime = FakeRuntime(next_ids=[1])
    engine = _engine(runtime)

    with pytest.raises(UnsupportedModelError, match="other-model"):
        asyncio.run(engine.generate(model_name="other-model", text="Hello"))

    assert runtime.load_calls == 0


def test_greedy_appends_argmax_and_drops_the_prompt() -> None:
    prompt_ids = [1, 2, 3]
    runtime = FakeRuntime(next_ids=[7])
    engine = _engine(runtime)
    engine.tokenizer = FakeTokenizer(prompt_ids, eos_token_id=None)  # type: ignore[assignment]
    engine.tokenizer_name = DEFAULT_MODEL

    result = asyncio.run(
        engine.generate(
            model_name=DEFAULT_MODEL,
            text="Hello",
            max_tokens=1,
            temperature=1,
        )
    )

    assert result.completion_token_ids == (7,)
    assert set(prompt_ids).isdisjoint(result.completion_token_ids)
    assert result.generated_text == "7"
    assert result.finish_reason == "length"


def test_eos_stops_early_including_on_the_last_step() -> None:
    runtime = FakeRuntime(next_ids=[4, 9, 3])
    tokenizer = FakeTokenizer([1], eos_token_id=9)
    engine = _engine(runtime)
    engine.tokenizer = tokenizer  # type: ignore[assignment]
    engine.tokenizer_name = DEFAULT_MODEL

    result = asyncio.run(
        engine.generate(model_name=DEFAULT_MODEL, text="Hello", max_tokens=5)
    )

    assert result.completion_token_ids == (4, 9)
    assert result.finish_reason == "stop"

    last_step = FakeRuntime(next_ids=[9])
    last_engine = _engine(last_step)
    last_engine.tokenizer = tokenizer  # type: ignore[assignment]
    last_engine.tokenizer_name = DEFAULT_MODEL
    last_result = asyncio.run(
        last_engine.generate(model_name=DEFAULT_MODEL, text="Hello", max_tokens=1)
    )
    assert last_result.completion_token_ids == (9,)
    assert last_result.finish_reason == "stop"


def test_missing_eos_runs_until_max_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticks = iter([100.0, 100.25])
    monkeypatch.setattr("glassbox.engine.engine.time.perf_counter", lambda: next(ticks))
    prompt_ids = [8, 6]
    runtime = FakeRuntime(next_ids=[3, 4, 5])
    tokenizer = FakeTokenizer(prompt_ids, eos_token_id=None)
    engine = _engine(runtime)
    engine.tokenizer = tokenizer  # type: ignore[assignment]
    engine.tokenizer_name = DEFAULT_MODEL

    result = asyncio.run(
        engine.generate(model_name=DEFAULT_MODEL, text="Hello", max_tokens=3)
    )

    assert result.completion_token_ids == (3, 4, 5)
    assert result.finish_reason == "length"
    assert result.prompt_tokens == len(prompt_ids)
    assert result.completion_tokens == 3
    assert result.latency_ms == 250
    assert result.tokens_per_second == 12


def test_second_generate_reuses_the_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = FakeTokenizer([1], eos_token_id=None)
    loads: list[str] = []

    def fake_from_pretrained(model_name: str) -> FakeTokenizer:
        loads.append(model_name)
        return tokenizer

    monkeypatch.setattr(
        "glassbox.engine.engine.AutoTokenizer.from_pretrained",
        fake_from_pretrained,
    )
    runtime = FakeRuntime(next_ids=[2, 2])
    engine = _engine(runtime)

    asyncio.run(engine.generate(model_name=DEFAULT_MODEL, text="Hello", max_tokens=1))
    asyncio.run(engine.generate(model_name=DEFAULT_MODEL, text="Hello", max_tokens=1))

    assert loads == [DEFAULT_MODEL]
    assert runtime.load_calls == 2
