from unittest.mock import MagicMock, patch

import pytest

from llm import LLMError, OpenRouterClient
from llm.base_client import LLMClient


def _make_client(response: str = "hello") -> tuple[OpenRouterClient, MagicMock]:
    mock_openai = MagicMock()
    mock_openai.OpenAI.return_value.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=response))]
    )
    with patch.dict("sys.modules", {"openai": mock_openai}):
        client = OpenRouterClient(api_key="sk-or-test", model_id="openai/gpt-4o-mini")
    client._client = mock_openai.OpenAI.return_value
    return client, mock_openai


def test_is_llm_client_subclass():
    assert issubclass(OpenRouterClient, LLMClient)


def test_complete_returns_string():
    client, _ = _make_client("openrouter response")
    assert client.complete("hello") == "openrouter response"


def test_complete_includes_system_message():
    client, mock_openai = _make_client()
    client.complete("user msg", system="you are helpful")
    call_messages = client._client.chat.completions.create.call_args[1]["messages"]
    assert call_messages[0] == {"role": "system", "content": "you are helpful"}
    assert call_messages[1]["role"] == "user"


def test_complete_no_system_omits_system_message():
    client, _ = _make_client()
    client.complete("user msg")
    call_messages = client._client.chat.completions.create.call_args[1]["messages"]
    assert all(m["role"] != "system" for m in call_messages)


def test_uses_openrouter_base_url():
    mock_openai = MagicMock()
    mock_openai.OpenAI.return_value.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="ok"))]
    )
    with patch.dict("sys.modules", {"openai": mock_openai}):
        OpenRouterClient(api_key="sk-or-test")
    _, kwargs = mock_openai.OpenAI.call_args
    assert "openrouter.ai" in kwargs.get("base_url", "")


def test_raises_llm_error_on_failure():
    client, _ = _make_client()
    client._client.chat.completions.create.side_effect = Exception("rate limit")
    with pytest.raises(LLMError, match="rate limit"):
        client.complete("prompt")


def test_extra_headers_sent():
    client, _ = _make_client()
    client.complete("hello")
    call_kwargs = client._client.chat.completions.create.call_args[1]
    assert "extra_headers" in call_kwargs
    assert "X-Title" in call_kwargs["extra_headers"]
