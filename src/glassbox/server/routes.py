import time
import uuid
from typing import Literal

from fastapi import APIRouter, HTTPException, Request

from glassbox.config import Settings
from glassbox.server.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    GlassboxMetrics,
    HealthResponse,
    ModelInfo,
    ModelListResponse,
    UsageInfo,
)


router = APIRouter()


def _dtype_for(device: Literal["cpu", "cuda"]) -> Literal["float32", "float16"]:
    return "float16" if device == "cuda" else "float32"


def _settings(request: Request) -> Settings:
    return request.app.state.settings


def _require_configured_model(model: str, settings: Settings) -> None:
    if model != settings.model:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model}' is not available",
        )


def _mock_usage(prompt: str, completion: str) -> UsageInfo:
    prompt_tokens = len(prompt.split())
    completion_tokens = len(completion.split())
    return UsageInfo(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


def _mock_metrics(
    started_at: float,
    usage: UsageInfo,
    settings: Settings,
) -> GlassboxMetrics:
    latency_ms = (time.perf_counter() - started_at) * 1000
    elapsed_seconds = max(latency_ms / 1000, 1e-9)
    return GlassboxMetrics(
        total_latency_ms=latency_ms,
        tokens_per_second=usage.completion_tokens / elapsed_seconds,
        device=settings.device,
        dtype=_dtype_for(settings.device),
        gpu_memory_allocated_mb=0,
        gpu_memory_reserved_mb=0,
    )


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    settings = _settings(request)
    return HealthResponse(
        status="alive",
        loaded=False,
        model=settings.model,
        device=settings.device,
        dtype=_dtype_for(settings.device),
        model_load_seconds=None,
    )


@router.get("/v1/models", response_model=ModelListResponse)
async def models(request: Request) -> ModelListResponse:
    settings = _settings(request)
    return ModelListResponse(
        data=[
            ModelInfo(
                id=settings.model,
                created=0,
            )
        ]
    )


@router.post("/v1/completions", response_model=CompletionResponse)
async def completions(
    payload: CompletionRequest,
    request: Request,
) -> CompletionResponse:
    started_at = time.perf_counter()
    settings = _settings(request)
    _require_configured_model(payload.model, settings)

    generated_text = "This is a simulated completion."
    usage = _mock_usage(payload.prompt, generated_text)

    return CompletionResponse(
        id=f"cmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=settings.model,
        choices=[
            CompletionResponseChoice(
                text=generated_text,
                index=0,
                logprobs=None,
                finish_reason="stop",
            )
        ],
        usage=usage,
        glassbox_metrics=_mock_metrics(started_at, usage, settings),
    )


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    payload: ChatCompletionRequest,
    request: Request,
) -> ChatCompletionResponse:
    started_at = time.perf_counter()
    settings = _settings(request)
    _require_configured_model(payload.model, settings)

    generated_text = "This is a simulated mock response"
    prompt = " ".join(message.content for message in payload.messages)
    usage = _mock_usage(prompt, generated_text)

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=settings.model,
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content=generated_text),
                finish_reason="stop",
            )
        ],
        usage=usage,
        glassbox_metrics=_mock_metrics(started_at, usage, settings),
    )
