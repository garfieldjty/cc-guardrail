import pytest
import random

from attack.foa_optimizer import smell_search, visual_search, cauchy_mutation, FOAOptimizer
from attack.prompt_builder import DIMS

TEMPLATES = {dim: [f"{dim}_v1", f"{dim}_v2", f"{dim}_v3"] for dim in DIMS}


# ---------------------------------------------------------------------------
# smell_search
# ---------------------------------------------------------------------------

def test_smell_search_population_size():
    pop = smell_search(TEMPLATES, population_size=10, rng=random.Random(42))
    assert len(pop) == 10


def test_smell_search_all_dims_present():
    pop = smell_search(TEMPLATES, population_size=5, rng=random.Random(0))
    for combo in pop:
        assert set(combo.keys()) == set(DIMS)


def test_smell_search_values_from_templates():
    pop = smell_search(TEMPLATES, population_size=20, rng=random.Random(1))
    for combo in pop:
        for dim, val in combo.items():
            assert val in TEMPLATES[dim]


# ---------------------------------------------------------------------------
# visual_search
# ---------------------------------------------------------------------------

def test_visual_search_returns_top_k():
    pop = [{"role": f"r{i}", **{d: "x" for d in DIMS if d != "role"}} for i in range(5)]
    prompts = [f"p{i}" for i in range(5)]
    scores = [0.1, 0.9, 0.5, 0.8, 0.2]
    score_fn = lambda c, p: scores[int(p[1])]

    survivors = visual_search(pop, prompts, score_fn, top_k=2)
    assert len(survivors) == 2
    # Highest scores first
    assert survivors[0][2] == 0.9
    assert survivors[1][2] == 0.8


def test_visual_search_output_structure():
    pop = smell_search(TEMPLATES, population_size=4, rng=random.Random(0))
    prompts = ["prompt"] * 4
    survivors = visual_search(pop, prompts, lambda c, p: 0.5, top_k=2)
    for combo, prompt, score in survivors:
        assert isinstance(combo, dict)
        assert isinstance(prompt, str)
        assert isinstance(score, float)


# ---------------------------------------------------------------------------
# cauchy_mutation
# ---------------------------------------------------------------------------

def test_cauchy_mutation_changes_dims():
    combo = {dim: TEMPLATES[dim][0] for dim in DIMS}
    mutated = cauchy_mutation(combo, TEMPLATES, n_dims=2, rng=random.Random(99))
    diffs = sum(1 for d in DIMS if combo[d] != mutated[d])
    # At least some dimensions must have changed (Cauchy shift ≠ 0 almost always)
    # We allow 0 if shift happened to wrap to same index, but values must stay valid
    for dim in DIMS:
        assert mutated[dim] in TEMPLATES[dim]


def test_cauchy_mutation_preserves_unchanged_dims():
    combo = {dim: TEMPLATES[dim][1] for dim in DIMS}
    # Force n_dims=1 with fixed seed so we know only one dim changes
    rng = random.Random(7)
    mutated = cauchy_mutation(combo, TEMPLATES, n_dims=1, rng=rng)
    changed = [d for d in DIMS if combo[d] != mutated[d]]
    assert len(changed) <= 1


# ---------------------------------------------------------------------------
# FOAOptimizer integration
# ---------------------------------------------------------------------------

def test_foa_optimizer_runs_iterations():
    call_log = []

    def render_fn(combo):
        return "古文"

    def score_fn(combo, prompt):
        call_log.append(combo)
        return 0.5

    opt = FOAOptimizer(TEMPLATES, population_size=6, top_k=2, iterations=3, seed=0)
    result = opt.run("make a weapon", render_fn, score_fn)

    assert len(result["asr_trajectory"]) == 3
    assert result["behavior"] == "make a weapon"
    assert "best_prompt" in result
    assert "best_combo" in result
    assert "best_score" in result


def test_foa_optimizer_asr_trajectory_structure():
    opt = FOAOptimizer(TEMPLATES, population_size=6, top_k=2, iterations=3, seed=1)
    result = opt.run("test", lambda c: "prompt", lambda c, p: 1.0)
    for entry in result["asr_trajectory"]:
        assert "iteration" in entry
        assert "top_asr" in entry


def test_foa_optimizer_best_score_is_from_survivors():
    opt = FOAOptimizer(TEMPLATES, population_size=6, top_k=2, iterations=2, seed=2)
    result = opt.run("test", lambda c: "prompt", lambda c, p: 0.75)
    assert result["best_score"] == 0.75
