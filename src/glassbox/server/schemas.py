from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from glassbox.config import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    MAX_ALLOWED_TOKENS,
)


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CompletionRequest(StrictRequest):
    model: str
    prompt: str
    max_tokens: int = Field(
        default=DEFAULT_MAX_TOKENS,
        ge=1,
        le=MAX_ALLOWED_TOKENS,
    )
    temperature: Literal[0] = DEFAULT_TEMPERATURE
    top_p: Literal[1] = 1
    n: Literal[1] = 1
    stream: Literal[False] = False
    stop: str | list[str] | None = None
    echo: Literal[False] = False


class CompletionResponseChoice(BaseModel):
    text: str
    index: int
    logprobs: None = None
    finish_reason: Literal["stop", "length"]


class GlassboxMetrics(BaseModel):
    total_latency_ms: float
    tokens_per_second: float
    device: Literal["cpu", "cuda"]
    dtype: Literal["float32", "float16"]
    gpu_memory_allocated_mb: float
    gpu_memory_reserved_mb: float


class CompletionResponse(BaseModel):
    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: list[CompletionResponseChoice]
    usage: UsageInfo
    glassbox_metrics: GlassboxMetrics


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class ChatCompletionRequest(StrictRequest):
    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: Literal[0] = DEFAULT_TEMPERATURE
    max_tokens: int = Field(
        default=DEFAULT_MAX_TOKENS,
        ge=1,
        le=MAX_ALLOWED_TOKENS,
    )
    top_p: Literal[1] = 1
    n: Literal[1] = 1
    stream: Literal[False] = False


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length"]


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionResponseChoice]
    usage: UsageInfo
    glassbox_metrics: GlassboxMetrics


class HealthResponse(BaseModel):
    status: Literal["alive"]
    loaded: bool
    model: str
    device: Literal["cpu", "cuda"]
    dtype: Literal["float32", "float16"]
    model_load_seconds: float | None


class ModelInfo(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: Literal["glassbox"] = "glassbox"


class ModelListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelInfo]
