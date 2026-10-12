import shutil
import subprocess
import threading
import time
from urllib.parse import urlparse

import requests

from kneevision.config import settings
from kneevision.config.settings import OLLAMA_MODEL, OLLAMA_URL

DEFAULT_OLLAMA_URL = OLLAMA_URL
DEFAULT_MODEL = OLLAMA_MODEL

_AUTOSTART_LOCK = threading.Lock()
_autostart_attempted = False


def _is_local(base_url: str) -> bool:
    """Local Ollama on its standard port only; any other host/port is the caller's own server."""
    u = urlparse(base_url)
    return u.hostname in ("localhost", "127.0.0.1", "::1") and (u.port or 80) == 11434


def start_local_ollama(base_url: str, wait_seconds: float = 20.0) -> bool:
    """Launch `ollama serve` detached and wait until it answers. Tries at most once per
    process; returns True if the server is reachable afterwards. Never raises."""
    global _autostart_attempted
    with _AUTOSTART_LOCK:
        if _autostart_attempted or not _is_local(base_url) or not shutil.which("ollama"):
            return False
        _autostart_attempted = True
        try:
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             stdin=subprocess.DEVNULL, start_new_session=True)
        except OSError:
            return False
        deadline = time.time() + wait_seconds
        while time.time() < deadline:
            try:
                requests.get(f"{base_url.rstrip('/')}/api/version", timeout=1).raise_for_status()
                return True
            except requests.exceptions.RequestException:
                time.sleep(0.5)
        return False


class LLMUnavailableError(RuntimeError):
    """Raised when the configured LLM (local or hosted) can't produce a response.
    Callers should catch this and fall back to a retrieval-only response rather
    than crashing the whole pipeline."""


class OllamaUnavailableError(LLMUnavailableError):
    """Raised when the local Ollama server can't be reached or the model
    isn't pulled."""


class OllamaClient:
    """Thin client for a local Ollama server (see https://ollama.com).
    Deliberately not an API-key-based cloud LLM: this project runs the
    synthesis step on a small local model instead. For hosted APIs see
    `OpenAICompatClient`.
    """
    provider = "ollama"

    @property
    def label(self) -> str:
        return f"{self.model} (local Ollama)"

    def __init__(self, model: str = DEFAULT_MODEL, base_url: str = DEFAULT_OLLAMA_URL, timeout: float = 60.0):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        try:
            try:
                resp = self._post(prompt)
            except requests.exceptions.ConnectionError:
                # Server not running. For a local Ollama, start it once and retry.
                if not (settings.OLLAMA_AUTOSTART and start_local_ollama(self.base_url)):
                    raise
                resp = self._post(prompt)
            resp.raise_for_status()
            # Parsing the body inside this try too: a 200 response with a
            # truncated/non-JSON body (proxy hiccup, server crash mid-write)
            # used to raise json.JSONDecodeError here, which is a ValueError,
            # not a RequestException -- it was propagating uncaught instead
            # of degrading gracefully like every other Ollama failure mode.
            data = resp.json()
        except (requests.exceptions.RequestException, ValueError) as exc:
            raise OllamaUnavailableError(
                f"Could not reach Ollama at {self.base_url} (is `ollama serve` running "
                f"and has `{self.model}` been pulled?): {exc}"
            ) from exc

        if "response" not in data:
            raise OllamaUnavailableError(f"Unexpected Ollama response: {data}")
        return data["response"].strip()

    def _post(self, prompt: str):
        return requests.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False, "options": {"temperature": 0}},
            timeout=self.timeout,
        )


class OpenAICompatClient:
    """Client for any OpenAI-compatible `/chat/completions` API, including xAI's Grok
    (https://docs.x.ai). The API key is passed in, never stored or logged here.

    Note: the prompt (guideline excerpts plus any patient context typed by the user) is
    sent to the provider. Callers should tell users not to enter identifying information.
    """

    def __init__(self, base_url: str, model: str, api_key: str, provider: str = "openai-compatible",
                 timeout: float = 60.0):
        if not api_key:
            raise LLMUnavailableError(f"No API key configured for provider '{provider}'.")
        self.base_url, self.model, self.provider = base_url.rstrip("/"), model, provider
        self._api_key, self.timeout = api_key, timeout

    @property
    def label(self) -> str:
        return f"{self.model} ({self.provider})"

    def generate(self, prompt: str) -> str:
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
                json={"model": self.model, "temperature": 0,
                      "messages": [{"role": "user", "content": prompt}]},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
        except (requests.exceptions.RequestException, ValueError, KeyError, IndexError, TypeError) as exc:
            # Don't echo the response body or headers: they can contain account details.
            status = getattr(getattr(exc, "response", None), "status_code", None)
            raise LLMUnavailableError(
                f"{self.provider} request failed ({type(exc).__name__}"
                f"{f', HTTP {status}' if status else ''})."
            ) from None
        if not isinstance(text, str) or not text.strip():
            raise LLMUnavailableError(f"{self.provider} returned an empty response.")
        return text.strip()


def make_llm_client(provider: str | None = None):
    """Build the LLM client from settings (env vars). Falls back to local Ollama when a
    hosted provider is requested but not configured, so the app degrades instead of failing."""
    provider = (provider or settings.LLM_PROVIDER).lower()
    if provider == "auto":
        provider = "xai" if settings.XAI_API_KEY else "ollama"
    if provider == "xai" and settings.XAI_API_KEY:
        return OpenAICompatClient(settings.XAI_BASE_URL, settings.XAI_MODEL, settings.XAI_API_KEY, provider="xAI Grok")
    if provider == "openai-compatible" and settings.LLM_API_KEY and settings.LLM_BASE_URL and settings.LLM_MODEL:
        return OpenAICompatClient(settings.LLM_BASE_URL, settings.LLM_MODEL, settings.LLM_API_KEY)
    return OllamaClient()
