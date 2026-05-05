from .base_client import LLMClient, LLMError


class OpenAIClient(LLMClient):
    def __init__(self, api_key: str, model_id: str = "gpt-4o-mini",
                 temperature: float = 1.0, max_tokens: int = 1024):
        super().__init__(model_id, temperature, max_tokens)
        try:
            import openai
            self._client = openai.OpenAI(api_key=api_key)
        except ImportError as e:
            raise LLMError("openai package not installed") from e

    def complete(self, prompt: str, system: str | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            resp = self._client.chat.completions.create(
                model=self.model_id, messages=messages,
                temperature=self.temperature, max_tokens=self.max_tokens,
            )
            return resp.choices[0].message.content
        except Exception as e:
            raise LLMError(str(e)) from e
