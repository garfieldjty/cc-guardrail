import time
import threading
from dataclasses import dataclass

from defense.lang_detector import detect_classical_chinese
from defense.intent_judge import IntentJudge


@dataclass
class GuardrailMetrics:
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


class Guardrail:
    def __init__(self, intent_judge: IntentJudge):
        self._judge = intent_judge
        self.metrics = GuardrailMetrics()
        self._metrics_lock = threading.Lock()

    def filter(self, prompt: str) -> tuple[bool, str]:
        """
        Returns (allowed, reason). Thread-safe — metrics updates are locked.
        allowed=False means the prompt was blocked.
        """
        t0 = time.monotonic()

        flagged, confidence = detect_classical_chinese(prompt)
        if not flagged:
            is_harmful, _ = self._judge.judge_direct(prompt)
            latency = (time.monotonic() - t0) * 1000
            with self._metrics_lock:
                self.metrics.total += 1
                self.metrics.total_latency_ms += latency
                if is_harmful:
                    self.metrics.blocked += 1
                else:
                    self.metrics.passed += 1
            if is_harmful:
                return False, "blocked: harmful intent detected (non-CC)"
            return True, "passed: not classical Chinese, safe"

        is_harmful, translated = self._judge.judge(prompt)
        latency = (time.monotonic() - t0) * 1000
        with self._metrics_lock:
            self.metrics.total += 1
            self.metrics.total_latency_ms += latency
            if is_harmful:
                self.metrics.blocked += 1
            else:
                self.metrics.passed += 1

        if is_harmful:
            return False, f"blocked: harmful intent detected (translation: {translated[:80]})"
        return True, "passed: classical Chinese but safe"
