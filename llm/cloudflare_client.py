import urllib.request
import urllib.error
import json
import logging

from .base_client import LLMClient, LLMError

logger = logging.getLogger(__name__)

class CloudflareClient(LLMClient):
    """Calls Cloudflare Workers AI via the REST API (no extra SDK needed)."""

    BASE_URL = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"

    def __init__(
        self,
        account_id: str,
        api_token: str,
        model_id: str = "@cf/meta/llama-3.1-8b-instruct",
        temperature: float = 1.0,
        max_tokens: int = 1024,
    ):
        super().__init__(model_id, temperature, max_tokens)
        self.account_id = account_id
        self.api_token = api_token

    def complete(self, prompt: str, system: str | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = json.dumps({
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }).encode()

        url = self.BASE_URL.format(account_id=self.account_id, model=self.model_id)
        logger.debug(f"Cloudflare API request to {url} with payload: {payload[:200]!r}")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req) as resp:
                body = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            raise LLMError(f"Cloudflare API error {e.code}: {e.read().decode()}") from e
        except Exception as e:
            raise LLMError(f"Request failed: {e}") from e

        if not body.get("success"):
            errors = body.get("errors", [])
            raise LLMError(f"Cloudflare returned errors: {errors}")

        result = body.get("result") or {}
        # Cloudflare native format: {"result": {"response": "..."}}
        if "response" in result:
            return result["response"]
        # OpenAI-compatible format: {"result": {"choices": [{"message": {"content": "..."}}]}}
        try:
            return result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            pass
        raise LLMError(f"Unexpected response shape: {body}")
