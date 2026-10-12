"""Hosted-LLM client (xAI Grok / OpenAI-compatible) and provider selection. HTTP is mocked."""
import pytest
import requests

from kneevision.config import settings
from kneevision.rag import llm as llm_module
from kneevision.rag.llm import LLMUnavailableError, OllamaClient, OpenAICompatClient, make_llm_client


class FakeResp:
    def __init__(self, payload=None, status=200):
        self.payload, self.status_code = payload, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code} secret-body", response=self)

    def json(self):
        return self.payload


def client():
    return OpenAICompatClient("https://api.example/v1/", "grok-test", "sk-secret", provider="xAI Grok")


def test_generate_posts_chat_completion_and_returns_text(monkeypatch):
    seen = {}

    def fake_post(url, headers, json, timeout):
        seen.update(url=url, headers=headers, body=json)
        return FakeResp({"choices": [{"message": {"content": "  Exercise helps.  "}}]})

    monkeypatch.setattr(llm_module.requests, "post", fake_post)
    assert client().generate("hi") == "Exercise helps."
    assert seen["url"] == "https://api.example/v1/chat/completions"
    assert seen["headers"]["Authorization"] == "Bearer sk-secret"
    assert seen["body"]["model"] == "grok-test" and seen["body"]["temperature"] == 0
    assert seen["body"]["messages"] == [{"role": "user", "content": "hi"}]


@pytest.mark.parametrize("resp", [
    FakeResp(status=401), FakeResp({"choices": []}), FakeResp({"unexpected": 1}),
    FakeResp({"choices": [{"message": {"content": "   "}}]}),
])
def test_bad_responses_raise_unavailable_without_leaking_secrets(monkeypatch, resp):
    monkeypatch.setattr(llm_module.requests, "post", lambda *a, **k: resp)
    with pytest.raises(LLMUnavailableError) as exc:
        client().generate("hi")
    assert "sk-secret" not in str(exc.value) and "secret-body" not in str(exc.value)


def test_network_error_raises_unavailable(monkeypatch):
    def boom(*a, **k):
        raise requests.exceptions.ConnectTimeout("t")
    monkeypatch.setattr(llm_module.requests, "post", boom)
    with pytest.raises(LLMUnavailableError):
        client().generate("hi")


def test_missing_key_is_rejected():
    with pytest.raises(LLMUnavailableError):
        OpenAICompatClient("https://x", "m", "")


def test_factory_defaults_to_ollama(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "ollama")
    assert isinstance(make_llm_client(), OllamaClient)


def test_factory_xai_with_key(monkeypatch):
    monkeypatch.setattr(settings, "XAI_API_KEY", "k")
    c = make_llm_client("xai")
    assert isinstance(c, OpenAICompatClient)
    assert "xAI" in c.label


def test_factory_auto_uses_xai_only_when_key_present(monkeypatch):
    monkeypatch.setattr(settings, "XAI_API_KEY", "")
    assert isinstance(make_llm_client("auto"), OllamaClient)
    monkeypatch.setattr(settings, "XAI_API_KEY", "k")
    assert isinstance(make_llm_client("auto"), OpenAICompatClient)


def test_factory_falls_back_to_ollama_when_hosted_not_configured(monkeypatch):
    monkeypatch.setattr(settings, "XAI_API_KEY", "")
    assert isinstance(make_llm_client("xai"), OllamaClient)
    monkeypatch.setattr(settings, "LLM_API_KEY", "")
    assert isinstance(make_llm_client("openai-compatible"), OllamaClient)


def test_pipeline_falls_back_to_excerpts_when_hosted_llm_fails(tmp_path, monkeypatch):
    from kneevision.rag import RehabRecommender
    (tmp_path / "kl2.md").write_text("---\nkl_grade: 2\ntopic: m\n---\n\n## Exercise\n\nStructured exercise is core.\n")
    monkeypatch.setattr(llm_module.requests, "post", lambda *a, **k: FakeResp(status=500))
    rec = RehabRecommender.from_guidelines_dir(tmp_path, llm=client()).recommend(2)
    assert not rec.used_llm and "Structured exercise" in rec.synthesis


# ── Ollama auto-start ──────────────────────────────────────────────────────────

def _reset_autostart(monkeypatch):
    monkeypatch.setattr(llm_module, "_autostart_attempted", False)
    monkeypatch.setattr(settings, "OLLAMA_AUTOSTART", True)


def test_ollama_autostarts_once_then_retries(monkeypatch):
    _reset_autostart(monkeypatch)
    calls = {"post": 0, "start": 0}

    def fake_post(url, json, timeout):
        calls["post"] += 1
        if calls["post"] == 1:
            raise requests.exceptions.ConnectionError("refused")
        return FakeResp({"response": " hello "})

    def fake_start(base_url, wait_seconds=20.0):
        calls["start"] += 1
        return True

    monkeypatch.setattr(llm_module.requests, "post", fake_post)
    monkeypatch.setattr(llm_module, "start_local_ollama", fake_start)
    assert OllamaClient(base_url="http://localhost:11434").generate("hi") == "hello"
    assert calls == {"post": 2, "start": 1}


def test_ollama_autostart_can_be_disabled(monkeypatch):
    _reset_autostart(monkeypatch)
    monkeypatch.setattr(settings, "OLLAMA_AUTOSTART", False)
    monkeypatch.setattr(llm_module, "start_local_ollama", lambda *a, **k: (_ for _ in ()).throw(AssertionError("started")))
    monkeypatch.setattr(llm_module.requests, "post", lambda *a, **k: (_ for _ in ()).throw(requests.exceptions.ConnectionError("x")))
    with pytest.raises(LLMUnavailableError):
        OllamaClient(base_url="http://localhost:11434").generate("hi")


def test_autostart_never_for_remote_hosts(monkeypatch):
    _reset_autostart(monkeypatch)
    monkeypatch.setattr(llm_module.shutil, "which", lambda _: "/usr/bin/ollama")
    spawned = []
    monkeypatch.setattr(llm_module.subprocess, "Popen", lambda *a, **k: spawned.append(a))
    assert llm_module.start_local_ollama("http://ollama.example.com:11434") is False
    assert llm_module.start_local_ollama("http://localhost:1") is False   # not the standard Ollama port
    assert spawned == []


def test_autostart_failure_falls_back_cleanly(monkeypatch):
    _reset_autostart(monkeypatch)
    monkeypatch.setattr(llm_module, "start_local_ollama", lambda *a, **k: False)
    monkeypatch.setattr(llm_module.requests, "post", lambda *a, **k: (_ for _ in ()).throw(requests.exceptions.ConnectionError("x")))
    with pytest.raises(LLMUnavailableError):
        OllamaClient(base_url="http://localhost:11434").generate("hi")
