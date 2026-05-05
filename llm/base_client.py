from abc import ABC, abstractmethod


class LLMError(Exception):
    pass


class LLMClient(ABC):
    def __init__(self, model_id: str, temperature: float = 1.0, max_tokens: int = 1024):
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def complete(self, prompt: str, system: str | None = None) -> str:
        """Send a prompt and return the response text. Raises LLMError on failure."""
