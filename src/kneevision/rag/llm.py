import requests

from kneevision.config.settings import OLLAMA_MODEL, OLLAMA_URL

DEFAULT_OLLAMA_URL = OLLAMA_URL
DEFAULT_MODEL = OLLAMA_MODEL


class OllamaUnavailableError(RuntimeError):
    """Raised when the local Ollama server can't be reached or the model
    isn't pulled. Callers should catch this and fall back to a
    retrieval-only response rather than crashing the whole pipeline."""


class OllamaClient:
    """Thin client for a local Ollama server (see https://ollama.com).
    Deliberately not an API-key-based cloud LLM: this project runs the
    synthesis step on a small local model instead.
    """

    def __init__(self, model: str = DEFAULT_MODEL, base_url: str = DEFAULT_OLLAMA_URL, timeout: float = 60.0):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.exceptions.RequestException as exc:
            raise OllamaUnavailableError(
                f"Could not reach Ollama at {self.base_url} (is `ollama serve` running "
                f"and has `{self.model}` been pulled?): {exc}"
            ) from exc

        data = resp.json()
        if "response" not in data:
            raise OllamaUnavailableError(f"Unexpected Ollama response: {data}")
        return data["response"].strip()
