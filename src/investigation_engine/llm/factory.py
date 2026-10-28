from dataclasses import dataclass, field

import httpx

from investigation_engine.llm.cache import CachedChatClient, ResponseCache
from investigation_engine.llm.client import (
    DEFAULT_RETRY_POLICY,
    ChatClient,
    OpenAiCompatibleClient,
    RetryPolicy,
)
from investigation_engine.llm.fallback import FallbackChatClient
from investigation_engine.settings import Settings

PROVIDER_NAMES = ("gemini", "groq")


class ConfigurationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ProviderSettings:
    name: str
    base_url: str
    api_key: str = field(repr=False)
    model: str


def configured_providers(settings: Settings) -> dict[str, ProviderSettings]:
    providers: dict[str, ProviderSettings] = {}
    if settings.gemini_api_key is not None:
        providers["gemini"] = ProviderSettings(
            "gemini",
            settings.gemini_base_url,
            settings.gemini_api_key.get_secret_value(),
            settings.gemini_model,
        )
    if settings.groq_api_key is not None:
        providers["groq"] = ProviderSettings(
            "groq",
            settings.groq_base_url,
            settings.groq_api_key.get_secret_value(),
            settings.groq_model,
        )
    return providers


def select_providers(settings: Settings, provider: str | None) -> list[ProviderSettings]:
    available = configured_providers(settings)
    if provider is None:
        if not available:
            raise ConfigurationError(
                "no LLM provider is configured: set GEMINI_API_KEY or GROQ_API_KEY"
            )
        return [available[name] for name in PROVIDER_NAMES if name in available]
    if provider not in PROVIDER_NAMES:
        raise ConfigurationError(
            f"unknown provider '{provider}', choose one of: {', '.join(PROVIDER_NAMES)}"
        )
    if provider not in available:
        raise ConfigurationError(f"{provider} is not configured: set {provider.upper()}_API_KEY")
    return [available[provider]]


def build_http_client(settings: Settings) -> httpx.Client:
    return httpx.Client(timeout=settings.llm_timeout_seconds)


def build_chat_client(
    settings: Settings,
    http: httpx.Client,
    cache: ResponseCache,
    provider: str | None = None,
    retry: RetryPolicy = DEFAULT_RETRY_POLICY,
) -> ChatClient:
    clients: list[ChatClient] = [
        CachedChatClient(
            OpenAiCompatibleClient(
                base_url=chosen.base_url,
                api_key=chosen.api_key,
                model=chosen.model,
                http=http,
                retry=retry,
            ),
            cache,
        )
        for chosen in select_providers(settings, provider)
    ]
    return clients[0] if len(clients) == 1 else FallbackChatClient(clients)
