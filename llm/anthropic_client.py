from .base_client import LLMClient, LLMError


class AnthropicClient(LLMClient):
    def __init__(self, api_key: str, model_id: str = "claude-haiku-4-5-20251001",
                 temperature: float = 1.0, max_tokens: int = 1024):
        super().__init__(model_id, temperature, max_tokens)
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key)
        except ImportError as e:
            raise LLMError("anthropic package not installed") from e

    def complete(self, prompt: str, system: str | None = None) -> str:
        kwargs = {"model": self.model_id, "max_tokens": self.max_tokens,
                  "messages": [{"role": "user", "content": prompt}]}
        if system:
            kwargs["system"] = system
        try:
            msg = self._client.messages.create(**kwargs)
            return msg.content[0].text
        except Exception as e:
            raise LLMError(str(e)) from e
