from .base_client import LLMClient, LLMError

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterClient(LLMClient):
    """OpenRouter via the OpenAI-compatible API."""

    def __init__(
        self,
        api_key: str,
        model_id: str = "openai/gpt-4o-mini",
        temperature: float = 1.0,
        max_tokens: int = 1024,
        site_url: str = "",
        site_name: str = "cc-guardrail",
    ):
        super().__init__(model_id, temperature, max_tokens)
        try:
            import openai
            self._client = openai.OpenAI(
                api_key=api_key,
                base_url=OPENROUTER_BASE_URL,
            )
        except ImportError as e:
            raise LLMError("openai package not installed") from e
        # OpenRouter recommends passing these headers for rate-limit tracking
        self._extra_headers = {
            **({"HTTP-Referer": site_url} if site_url else {}),
            "X-Title": site_name,
        }

    def complete(self, prompt: str, system: str | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            resp = self._client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                extra_headers=self._extra_headers,
            )
            return resp.choices[0].message.content
        except Exception as e:
            raise LLMError(str(e)) from e
