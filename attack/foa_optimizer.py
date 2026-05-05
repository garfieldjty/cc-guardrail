import random
import logging
import math
from typing import Callable

from attack.prompt_builder import DIMS
from utils.concurrency import interruptible_map

logger = logging.getLogger(__name__)

# Score function type: (combo, cc_prompt) -> float in [0, 1]
ScoreFn = Callable[[dict[str, str], str], float]


def smell_search(
    templates: dict[str, list[str]], population_size: int, rng: random.Random | None = None
) -> list[dict[str, str]]:
    """Generate an initial population by randomly sampling across all 8 dimensions."""
    rng = rng or random.Random()
    return [
        {dim: rng.choice(templates[dim]) for dim in DIMS}
        for _ in range(population_size)
    ]


def visual_search(
    population: list[dict[str, str]],
    prompts: list[str],
    score_fn: ScoreFn,
    top_k: int,
) -> list[tuple[dict[str, str], str, float]]:
    """Score each candidate sequentially, return the top_k survivors."""
    scored = [(combo, prompt, score_fn(combo, prompt)) for combo, prompt in zip(population, prompts)]
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:top_k]


def cauchy_mutation(
    combo: dict[str, str],
    templates: dict[str, list[str]],
    n_dims: int = 2,
    rng: random.Random | None = None,
) -> dict[str, str]:
    """Perturb a combo by replacing n_dims dimensions using a Cauchy-inspired index shift."""
    rng = rng or random.Random()
    mutated = dict(combo)
    dims_to_mutate = rng.sample(DIMS, k=min(n_dims, len(DIMS)))
    for dim in dims_to_mutate:
        pool = templates[dim]
        current_idx = pool.index(mutated[dim]) if mutated[dim] in pool else 0
        shift = int(round(math.tan(math.pi * (rng.random() - 0.5))))
        new_idx = (current_idx + shift) % len(pool)
        mutated[dim] = pool[new_idx]
    return mutated


def _parallel_render(render_fn, population: list[dict], max_workers: int) -> list[str]:
    """Render all combos concurrently; preserves order."""
    return interruptible_map(render_fn, population, max_workers)


def _parallel_score(
    population: list[dict], prompts: list[str], score_fn: ScoreFn, max_workers: int
) -> list[tuple[dict, str, float]]:
    """Score all (combo, prompt) pairs concurrently; preserves order."""
    pairs = list(zip(population, prompts))

    def _score(pair):
        combo, prompt = pair
        return combo, prompt, score_fn(combo, prompt)

    scored = interruptible_map(_score, pairs, max_workers)
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored


class FOAOptimizer:
    def __init__(
        self,
        templates: dict[str, list[str]],
        population_size: int = 10,
        top_k: int = 3,
        iterations: int = 3,
        seed: int | None = None,
        max_workers: int = 5,
    ):
        self._templates = templates
        self._population_size = population_size
        self._top_k = top_k
        self._iterations = iterations
        self._rng = random.Random(seed)
        self._max_workers = max_workers

    def run(
        self,
        behavior: str,
        render_fn: Callable[[dict[str, str]], str],
        score_fn: ScoreFn,
    ) -> dict:
        """
        Run FOA for one behavior with concurrent rendering and scoring.
        render_fn: (combo) -> classical_chinese_prompt string
        score_fn:  (combo, prompt) -> float in [0,1]
        """
        trajectory = []

        logger.debug(f"FOA smell search — population={self._population_size}  workers={self._max_workers}")
        population = smell_search(self._templates, self._population_size, self._rng)
        prompts = _parallel_render(render_fn, population, self._max_workers)

        survivors = None
        for iteration in range(self._iterations):
            logger.debug(f"FOA iteration {iteration + 1}/{self._iterations} — scoring {len(population)} candidates")
            scored = _parallel_score(population, prompts, score_fn, self._max_workers)
            survivors = scored[:self._top_k]
            asr = sum(s for _, _, s in survivors) / len(survivors) if survivors else 0.0
            trajectory.append({"iteration": iteration + 1, "top_asr": asr})
            logger.debug(f"FOA iteration {iteration + 1} top-k ASR={asr:.2f}  scores={[round(s, 2) for _, _, s in survivors]}")

            if iteration < self._iterations - 1:
                # Generate next population from survivors via Cauchy mutation (sequential —
                # rng state must stay deterministic)
                new_combos = []
                for combo, _, _ in survivors:
                    for _ in range(self._population_size // self._top_k):
                        new_combos.append(cauchy_mutation(combo, self._templates, rng=self._rng))
                population = new_combos
                prompts = _parallel_render(render_fn, population, self._max_workers)

        best_combo, best_prompt, best_score = survivors[0]
        return {
            "behavior": behavior,
            "best_combo": best_combo,
            "best_prompt": best_prompt,
            "best_score": best_score,
            "asr_trajectory": trajectory,
        }
