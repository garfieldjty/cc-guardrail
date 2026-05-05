import time
import threading
from dataclasses import dataclass

from llm.base_client import LLMClient, LLMError

# Llama Guard 4 evaluation prompt — wraps the user prompt in the expected format.
# The model responds with "safe" or "unsafe\n<category list>".
_EVAL_TEMPLATE = """\
<BEGIN CONVERSATION>
User: {prompt}
<END CONVERSATION>

Provide your safety assessment for ONLY THE LAST User message in the above conversation:
 - First line must read 'safe' or 'unsafe'.
 - If unsafe, a second line must include a comma-separated list of violated categories."""


@dataclass
class LlamaGuardMetrics:
    total: int = 0
    blocked: int = 0
    passed: int = 0
    total_latency_ms: float = 0.0

    @property
    def block_rate(self) -> float:
        return self.blocked / self.total if self.total else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.total if self.total else 0.0


class LlamaGuardGuardrail:
    """Guardrail backed by meta-llama/llama-guard-4-12b via OpenRouter."""

    def __init__(self, client: LLMClient):
        self._client = client
        self.metrics = LlamaGuardMetrics()
        self._lock = threading.Lock()

    def filter(self, prompt: str) -> tuple[bool, str]:
        """Returns (allowed, reason). Thread-safe."""
        t0 = time.monotonic()

        try:
            message = _EVAL_TEMPLATE.format(prompt=prompt)
            response = (self._client.complete(message) or "").strip()
            first_line = response.split("\n")[0].strip().lower()
            is_unsafe = first_line == "unsafe"
            category = response.split("\n")[1].strip() if is_unsafe and "\n" in response else ""
        except LLMError as e:
            is_unsafe = False
            category = f"error: {str(e)[:60]}"

        latency = (time.monotonic() - t0) * 1000
        with self._lock:
            self.metrics.total += 1
            self.metrics.total_latency_ms += latency
            if is_unsafe:
                self.metrics.blocked += 1
            else:
                self.metrics.passed += 1

        if is_unsafe:
            return False, f"blocked: llama_guard unsafe ({category})"
        return True, "passed: llama_guard safe"
