"""Runtime settings, read from environment variables and an optional ``.env`` file."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")

# Supported model providers. "openai" also covers any OpenAI-compatible endpoint
# (OpenRouter, Groq, DeepSeek, LM Studio, vLLM, …) via LEGAL_ANALYZER_OPENAI_BASE_URL.
PROVIDERS = ("anthropic", "openai", "gemini", "ollama")
PROVIDER_LABELS = {
    "anthropic": "Anthropic Claude",
    "openai": "OpenAI",
    "gemini": "Google Gemini",
    "ollama": "Ollama (local)",
}
DEFAULT_MODELS = {
    "anthropic": "claude-opus-5-5",
    "openai": "gpt-5",
    "gemini": "gemini-flash-latest",
    "ollama": "qwen3.5:4b",
}
# Claude models offered in the web UI's model picker (the configured model is always included).
ANTHROPIC_MODELS = ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# List prices in USD per million tokens. Used only for the cost *estimate* shown in reports;
# models not listed here (and local models) show no estimate or a cost of zero.
PRICING: dict[str, dict[str, float]] = {
    "claude-opus-5-5": {"input": 4.00, "output": 20.00, "cache_write": 5.00, "cache_read": 0.20},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00, "cache_write": 2.50, "cache_read": 0.20},
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50, "cache_write": 0.125, "cache_read": 0.01},
}


def load_dotenv(path: Path) -> None:
    """Load ``KEY=VALUE`` lines from a .env file. Variables already set in the environment win."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


def _effort(value: str, name: str) -> str:
    value = value.strip().lower()
    if value not in EFFORT_LEVELS:
        raise ValueError(f"{name} must be one of {', '.join(EFFORT_LEVELS)} (got {value!r})")
    return value


def _provider(value: str) -> str:
    value = value.strip().lower()
    if value not in PROVIDERS:
        raise ValueError(f"provider must be one of {', '.join(PROVIDERS)} (got {value!r})")
    return value


def _flag(value: str) -> bool:
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _optional(value: str | None) -> str | None:
    if value is None:
        return None
    return value.strip() or None


@dataclass(frozen=True)
class Settings:
    provider: str = "anthropic"
    # Overrides the model of the default provider (LEGAL_ANALYZER_MODEL / --model).
    model: str | None = None
    anthropic_model: str = DEFAULT_MODELS["anthropic"]
    openai_model: str = DEFAULT_MODELS["openai"]
    gemini_model: str = DEFAULT_MODELS["gemini"]
    ollama_model: str = DEFAULT_MODELS["ollama"]

    # Effort controls how much Claude thinks. Clause analysis is the intelligence-sensitive step.
    analysis_effort: str = "high"
    synthesis_effort: str = "medium"
    chat_effort: str = "medium"
    # Server-side refusal fallback (Claude): if a safety classifier declines a request, the API
    # retries it on Anthropic's recommended fallback model instead of failing.
    use_fallbacks: bool = True

    # OpenAI-compatible endpoints
    openai_base_url: str | None = None
    openai_reasoning_effort: str | None = None  # sent only when set, for reasoning models
    context_window: int | None = None  # override the assumed context window of cloud models

    # Ollama
    ollama_host: str | None = None  # None: OLLAMA_HOST or http://localhost:11434
    ollama_num_ctx: int = 16_384
    ollama_num_predict: int = 4_096
    ollama_think: str | None = None  # None/"false": off; "true"/"low"/"medium"/"high": on
    ollama_keep_alive: str = "10m"
    ollama_parallel: int = 1

    data_dir: Path = Path("data")
    embedding_backend: str = "fastembed"  # "fastembed" (hybrid BM25 + dense) or "none" (BM25 only)
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    concurrency: int = 4
    max_retries: int = 4

    @classmethod
    def from_env(cls, env_file: Path | None = Path(".env")) -> Settings:
        if env_file is not None:
            load_dotenv(env_file)
        env = os.environ.get
        return cls(
            provider=_provider(env("LEGAL_ANALYZER_PROVIDER", "anthropic")),
            model=_optional(env("LEGAL_ANALYZER_MODEL")),
            anthropic_model=env("LEGAL_ANALYZER_ANTHROPIC_MODEL", DEFAULT_MODELS["anthropic"]),
            openai_model=env("LEGAL_ANALYZER_OPENAI_MODEL", DEFAULT_MODELS["openai"]),
            gemini_model=env("LEGAL_ANALYZER_GEMINI_MODEL", DEFAULT_MODELS["gemini"]),
            ollama_model=env("LEGAL_ANALYZER_OLLAMA_MODEL", DEFAULT_MODELS["ollama"]),
            analysis_effort=_effort(env("LEGAL_ANALYZER_EFFORT", "high"), "LEGAL_ANALYZER_EFFORT"),
            synthesis_effort=_effort(env("LEGAL_ANALYZER_SYNTHESIS_EFFORT", "medium"), "LEGAL_ANALYZER_SYNTHESIS_EFFORT"),
            chat_effort=_effort(env("LEGAL_ANALYZER_CHAT_EFFORT", "medium"), "LEGAL_ANALYZER_CHAT_EFFORT"),
            use_fallbacks=_flag(env("LEGAL_ANALYZER_FALLBACKS", "on")),
            openai_base_url=_optional(env("LEGAL_ANALYZER_OPENAI_BASE_URL")),
            openai_reasoning_effort=_optional(env("LEGAL_ANALYZER_OPENAI_REASONING_EFFORT")),
            context_window=int(env("LEGAL_ANALYZER_CONTEXT_WINDOW")) if env("LEGAL_ANALYZER_CONTEXT_WINDOW") else None,
            ollama_host=_optional(env("LEGAL_ANALYZER_OLLAMA_HOST")),
            ollama_num_ctx=max(2048, int(env("LEGAL_ANALYZER_OLLAMA_NUM_CTX", "16384"))),
            ollama_num_predict=max(512, int(env("LEGAL_ANALYZER_OLLAMA_NUM_PREDICT", "4096"))),
            ollama_think=_optional(env("LEGAL_ANALYZER_OLLAMA_THINK")),
            ollama_keep_alive=env("LEGAL_ANALYZER_OLLAMA_KEEP_ALIVE", "10m"),
            ollama_parallel=max(1, int(env("LEGAL_ANALYZER_OLLAMA_PARALLEL", "1"))),
            data_dir=Path(env("LEGAL_ANALYZER_DATA_DIR", "data")).expanduser(),
            embedding_backend=env("LEGAL_ANALYZER_EMBEDDINGS", "fastembed").strip().lower(),
            embedding_model=env("LEGAL_ANALYZER_EMBED_MODEL", "BAAI/bge-small-en-v1.5"),
            concurrency=max(1, int(env("LEGAL_ANALYZER_CONCURRENCY", "4"))),
            max_retries=max(0, int(env("LEGAL_ANALYZER_MAX_RETRIES", "4"))),
        )

    def with_overrides(self, **changes) -> Settings:
        clean = {k: v for k, v in changes.items() if v is not None}
        for key in ("analysis_effort", "synthesis_effort", "chat_effort"):
            if key in clean:
                clean[key] = _effort(clean[key], key)
        if "provider" in clean:
            clean["provider"] = _provider(clean["provider"])
        return replace(self, **clean)

    def model_for(self, provider: str) -> str:
        """The model to use for ``provider`` when none is chosen explicitly."""
        if self.model and provider == self.provider:
            return self.model
        return {
            "anthropic": self.anthropic_model,
            "openai": self.openai_model,
            "gemini": self.gemini_model,
            "ollama": self.ollama_model,
        }[_provider(provider)]

    @property
    def documents_dir(self) -> Path:
        return self.data_dir / "documents"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"
