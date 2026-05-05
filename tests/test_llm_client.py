import json
import urllib.error
from unittest.mock import patch, MagicMock

import pytest

from llm import LLMClient, LLMError, CloudflareClient, AnthropicClient, OpenAIClient


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------

def test_llm_client_is_abstract():
    with pytest.raises(TypeError):
        LLMClient(model_id="x")  # type: ignore


# ---------------------------------------------------------------------------
# CloudflareClient
# ---------------------------------------------------------------------------

CF_RESPONSE = json.dumps({
    "success": True,
    "result": {"response": "hello from CF"},
    "errors": [],
}).encode()


def _make_cf_client():
    return CloudflareClient(account_id="acct123", api_token="tok456")


def _mock_urlopen(response_bytes: bytes):
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=MagicMock(read=MagicMock(return_value=response_bytes)))
    cm.__exit__ = MagicMock(return_value=False)
    return cm


@patch("llm.cloudflare_client.urllib.request.urlopen")
def test_cf_complete_returns_string(mock_urlopen):
    mock_urlopen.return_value = _mock_urlopen(CF_RESPONSE)
    client = _make_cf_client()
    result = client.complete("tell me something")
    assert result == "hello from CF"


@patch("llm.cloudflare_client.urllib.request.urlopen")
def test_cf_complete_includes_system(mock_urlopen):
    mock_urlopen.return_value = _mock_urlopen(CF_RESPONSE)
    client = _make_cf_client()
    client.complete("user prompt", system="you are helpful")
    call_args = mock_urlopen.call_args[0][0]
    body = json.loads(call_args.data)
    assert body["messages"][0] == {"role": "system", "content": "you are helpful"}
    assert body["messages"][1]["role"] == "user"


@patch("llm.cloudflare_client.urllib.request.urlopen")
def test_cf_raises_llm_error_on_http_error(mock_urlopen):
    mock_urlopen.side_effect = urllib.error.HTTPError(
        url="", code=500, msg="Internal", hdrs=None, fp=MagicMock(read=MagicMock(return_value=b"err"))
    )
    client = _make_cf_client()
    with pytest.raises(LLMError, match="500"):
        client.complete("prompt")


@patch("llm.cloudflare_client.urllib.request.urlopen")
def test_cf_raises_llm_error_on_api_failure(mock_urlopen):
    bad_body = json.dumps({"success": False, "errors": [{"message": "quota exceeded"}]}).encode()
    mock_urlopen.return_value = _mock_urlopen(bad_body)
    client = _make_cf_client()
    with pytest.raises(LLMError, match="quota exceeded"):
        client.complete("prompt")


# ---------------------------------------------------------------------------
# AnthropicClient
# ---------------------------------------------------------------------------

def test_anthropic_complete_returns_string(mocker):
    mock_anthropic = MagicMock()
    mock_anthropic.Anthropic.return_value.messages.create.return_value = MagicMock(
        content=[MagicMock(text="anthropic response")]
    )
    mocker.patch.dict("sys.modules", {"anthropic": mock_anthropic})

    client = AnthropicClient(api_key="key")
    result = client.complete("hello")
    assert result == "anthropic response"


def test_anthropic_raises_llm_error_on_failure(mocker):
    mock_anthropic = MagicMock()
    mock_anthropic.Anthropic.return_value.messages.create.side_effect = Exception("rate limit")
    mocker.patch.dict("sys.modules", {"anthropic": mock_anthropic})

    client = AnthropicClient(api_key="key")
    with pytest.raises(LLMError, match="rate limit"):
        client.complete("hello")


# ---------------------------------------------------------------------------
# OpenAIClient
# ---------------------------------------------------------------------------

def test_openai_complete_returns_string(mocker):
    mock_openai = MagicMock()
    mock_openai.OpenAI.return_value.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="openai response"))]
    )
    mocker.patch.dict("sys.modules", {"openai": mock_openai})

    client = OpenAIClient(api_key="key")
    result = client.complete("hello")
    assert result == "openai response"


def test_openai_raises_llm_error_on_failure(mocker):
    mock_openai = MagicMock()
    mock_openai.OpenAI.return_value.chat.completions.create.side_effect = Exception("timeout")
    mocker.patch.dict("sys.modules", {"openai": mock_openai})

    client = OpenAIClient(api_key="key")
    with pytest.raises(LLMError, match="timeout"):
        client.complete("hello")


# ---------------------------------------------------------------------------
# Interface contract: all clients are LLMClient subclasses
# ---------------------------------------------------------------------------

def test_all_clients_are_subclasses():
    assert issubclass(CloudflareClient, LLMClient)
    assert issubclass(AnthropicClient, LLMClient)
    assert issubclass(OpenAIClient, LLMClient)
