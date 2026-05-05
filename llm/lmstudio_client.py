from .base_client import LLMClient, LLMError

LMSTUDIO_DEFAULT_URL = "http://localhost:1234/v1"


class LMStudioClient(LLMClient):
    """Calls a local LM Studio server via its OpenAI-compatible REST API.

    Thinking is disabled by default via extra_body so that reasoning models
    (e.g. DeepSeek-R1) return only the final answer without <think> blocks.
    """

    def __init__(
        self,
        model_id: str = "local-model",
        base_url: str = LMSTUDIO_DEFAULT_URL,
        temperature: float = 1.0,
        max_tokens: int = 1024,
        enable_thinking: bool = False,
    ):
        super().__init__(model_id, temperature, max_tokens)
        self._enable_thinking = enable_thinking
        try:
            import openai
            self._client = openai.OpenAI(
                api_key="lm-studio",   # LM Studio accepts any non-empty string
                base_url=base_url,
            )
        except ImportError as e:
            raise LLMError("openai package not installed") from e

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
                extra_body={"enable_thinking": self._enable_thinking},
            )
            return resp.choices[0].message.content
        except Exception as e:
            raise LLMError(str(e)) from e
