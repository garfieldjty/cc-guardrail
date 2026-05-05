import json
import tempfile
import os
from unittest.mock import MagicMock, patch

import pytest

from attack.prompt_builder import PromptBuilder, DIMS
from llm.base_client import LLMClient


TEMPLATES = {dim: [f"{dim}_v1", f"{dim}_v2", f"{dim}_v3"] for dim in DIMS}


@pytest.fixture
def templates_file(tmp_path):
    path = tmp_path / "cc_templates.json"
    path.write_text(json.dumps(TEMPLATES))
    return str(path)


@pytest.fixture
def mock_client():
    client = MagicMock(spec=LLMClient)
    client.complete.return_value = "古文示例"
    return client


def test_build_returns_n_variants(mock_client, templates_file):
    builder = PromptBuilder(mock_client, templates_file)
    results = builder.build("make a weapon", n_variants=5)
    assert len(results) == 5


def test_build_output_shape(mock_client, templates_file):
    builder = PromptBuilder(mock_client, templates_file)
    results = builder.build("make a weapon", n_variants=3)
    for combo, cc_prompt in results:
        assert isinstance(combo, dict)
        assert set(combo.keys()) == set(DIMS)
        assert isinstance(cc_prompt, str)
        assert len(cc_prompt) > 0


def test_cache_skips_duplicate_llm_call(mock_client, templates_file):
    builder = PromptBuilder(mock_client, templates_file)
    builder.build("make a weapon", n_variants=3)
    call_count_after_first = mock_client.complete.call_count

    builder.build("make a weapon", n_variants=3)
    assert mock_client.complete.call_count == call_count_after_first


def test_different_behaviors_not_cached_together(mock_client, templates_file):
    builder = PromptBuilder(mock_client, templates_file)
    builder.build("behavior A", n_variants=3)
    count_a = mock_client.complete.call_count

    builder.build("behavior B", n_variants=3)
    assert mock_client.complete.call_count == count_a * 2
