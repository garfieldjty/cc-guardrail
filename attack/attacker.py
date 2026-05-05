import threading
import logging

from llm.base_client import LLMClient, LLMError
from attack.prompt_builder import PromptBuilder
from attack.foa_optimizer import FOAOptimizer
from eval.asr_judge import ASRJudge
from utils.concurrency import interruptible_map

logger = logging.getLogger(__name__)


class Attacker:
    def __init__(
        self,
        target_client: LLMClient,
        judge_client: LLMClient,
        prompt_builder: PromptBuilder,
        optimizer: FOAOptimizer,
        max_workers: int = 3,
        translator_client: LLMClient | None = None,
    ):
        self._target = target_client
        self._judge = ASRJudge(judge_client, translator=translator_client)
        self._builder = prompt_builder
        self._optimizer = optimizer
        self._max_workers = max_workers

    def _score(self, behavior: str, cc_prompt: str) -> tuple[float, str]:
        """
        Returns (normalized_fitness_score, response_text).
        Fitness score is normalized: 0-120 → 0.0-1.0
        """
        try:
            response = self._target.complete(cc_prompt) or ""
        except LLMError as e:
            logger.info(f"Target blocked: {str(e)[:180]}")
            return 0.0, f"[TARGET_BLOCKED: {e}]"
        
        if not response:
            logger.debug("Empty response from target")
            return 0.0, ""
        
        try:
            fitness_raw = self._judge.get_fitness_score(behavior, response)
            normalized_score = fitness_raw / ASRJudge.MAX_SCORE
            is_success = fitness_raw >= ASRJudge.SUCCESS_THRESHOLD
            status = "✓ HIT" if is_success else "✗ MISS"
            logger.info(
                f"{status} — fitness={fitness_raw}/120 (norm={normalized_score:.3f}) | "
                f"response={response[:200]!r}"
            )
            return normalized_score, response
        except LLMError as e:
            logger.error(f"Judge error: {e}")
            return 0.0, response

    def attack(self, behavior: str) -> dict:
        """
        Run CC-BOS attack for a single behavior.
        Rendering and scoring run concurrently inside the optimizer;
        successful_attacks is protected by a lock for thread safety.
        """
        successful_attacks: list[dict] = []
        lock = threading.Lock()

        def render_fn(combo):
            return self._builder.render(combo, behavior)

        success_threshold = ASRJudge.SUCCESS_THRESHOLD / ASRJudge.MAX_SCORE

        def score_fn(combo, prompt):
            score, response = self._score(behavior, prompt)
            if score >= success_threshold:
                with lock:
                    successful_attacks.append({
                        "prompt": prompt,
                        "response": response,
                        "score": score,
                        "combo": combo,
                    })
            return score

        result = self._optimizer.run(behavior, render_fn, score_fn)
        result["successful_attacks"] = successful_attacks

        try:
            result["response"] = self._target.complete(result["best_prompt"]) or ""
        except LLMError as e:
            result["response"] = f"[BLOCKED: {e}]"

        logger.info(
            f"Attack complete | best_fitness={result['best_score']:.3f} | "
            f"successful_attacks={len(successful_attacks)}"
        )
        return result

    def run_all(self, behaviors: list[dict]) -> list[dict]:
        """
        Run attack over all behaviors concurrently (behavior-level parallelism).
        Results are returned in the original order.
        """
        n = len(behaviors)
        results: list[dict] = [None] * n  # type: ignore

        def _attack_entry(args):
            idx, entry = args
            logger.info(f"[{idx + 1}/{n}] Attacking: {entry['behavior'][:70]}")
            r = self.attack(entry["behavior"])
            r["id"] = entry["id"]
            r["category"] = entry["category"]
            return idx, r

        for idx, r in interruptible_map(_attack_entry, list(enumerate(behaviors)), self._max_workers):
            results[idx] = r

        return results
