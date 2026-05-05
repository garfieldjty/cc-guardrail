from unittest.mock import MagicMock, patch

import pytest

from llm import LLMError, LMStudioClient
from llm.base_client import LLMClient
from llm.lmstudio_client import LMSTUDIO_DEFAULT_URL


def _make_client(response: str = "local response") -> tuple[LMStudioClient, MagicMock]:
    mock_openai = MagicMock()
    mock_openai.OpenAI.return_value.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=response))]
    )
    with patch.dict("sys.modules", {"openai": mock_openai}):
        client = LMStudioClient(model_id="local-model")
    client._client = mock_openai.OpenAI.return_value
    return client, mock_openai


def test_is_llm_client_subclass():
    assert issubclass(LMStudioClient, LLMClient)


def test_complete_returns_string():
    client, _ = _make_client("hello from lmstudio")
    assert client.complete("hi") == "hello from lmstudio"


def test_uses_local_base_url():
    mock_openai = MagicMock()
    mock_openai.OpenAI.return_value.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="ok"))]
    )
    with patch.dict("sys.modules", {"openai": mock_openai}):
        LMStudioClient()
    _, kwargs = mock_openai.OpenAI.call_args
    assert kwargs.get("base_url") == LMSTUDIO_DEFAULT_URL


def test_custom_base_url():
    mock_openai = MagicMock()
    mock_openai.OpenAI.return_value.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="ok"))]
    )
    with patch.dict("sys.modules", {"openai": mock_openai}):
        LMStudioClient(base_url="http://localhost:9999/v1")
    _, kwargs = mock_openai.OpenAI.call_args
    assert "9999" in kwargs.get("base_url", "")


def test_complete_with_system():
    client, _ = _make_client()
    client.complete("prompt", system="be concise")
    messages = client._client.chat.completions.create.call_args[1]["messages"]
    assert messages[0] == {"role": "system", "content": "be concise"}
    assert messages[1]["role"] == "user"


def test_raises_llm_error_on_failure():
    client, _ = _make_client()
    client._client.chat.completions.create.side_effect = Exception("connection refused")
    with pytest.raises(LLMError, match="connection refused"):
        client.complete("prompt")


def test_thinking_disabled_by_default():
    client, _ = _make_client()
    client.complete("prompt")
    call_kwargs = client._client.chat.completions.create.call_args[1]
    assert call_kwargs.get("extra_body", {}).get("enable_thinking") is False


def test_thinking_can_be_enabled():
    mock_openai = MagicMock()
    mock_openai.OpenAI.return_value.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="ok"))]
    )
    with patch.dict("sys.modules", {"openai": mock_openai}):
        client = LMStudioClient(enable_thinking=True)
    client._client = mock_openai.OpenAI.return_value
    client.complete("prompt")
    call_kwargs = client._client.chat.completions.create.call_args[1]
    assert call_kwargs["extra_body"]["enable_thinking"] is True


def test_default_temperature_is_1():
    client, _ = _make_client()
    assert client.temperature == 1.0
