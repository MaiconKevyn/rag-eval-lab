from __future__ import annotations

import json
from pathlib import Path

from scripts.generate_report import _build_comparisons, _comparison_key, _framework_from_experiment_id
from rag_eval_lab.tracking.mlflow_logger import _detect_framework


def test_framework_detection_helpers() -> None:
    assert _framework_from_experiment_id("exp_001_llamaindex") == "llamaindex"
    assert _framework_from_experiment_id("exp_001_chunk256_ada") == "vanilla"
    assert _comparison_key("exp_001_llamaindex") == "exp_001"
    assert _comparison_key("exp_001_chunk256_ada") == "exp_001"
    assert _detect_framework("exp_003_llamaindex") == "llamaindex"
    assert _detect_framework("exp_003_chunk128_topk10") == "vanilla"


def test_build_comparisons_pairs_vanilla_and_llamaindex() -> None:
    experiments = [
        {
            "experiment_id": "exp_001_chunk256_ada",
            "framework": "vanilla",
            "comparison_key": "exp_001",
            "chunk_size": 256,
            "top_k": 5,
            "score_threshold": 0.6,
            "composite_mean": 3.6,
            "empty_context_rate": 0.2,
            "total_cost_usd": 0.18,
        },
        {
            "experiment_id": "exp_001_llamaindex",
            "framework": "llamaindex",
            "comparison_key": "exp_001",
            "chunk_size": 256,
            "top_k": 5,
            "score_threshold": 0.6,
            "composite_mean": 3.5,
            "empty_context_rate": 0.22,
            "total_cost_usd": 0.01,
        },
    ]

    comparisons = _build_comparisons(experiments)
    assert len(comparisons) == 1
    assert comparisons[0]["comparison_key"] == "exp_001"
    assert comparisons[0]["delta_composite"] == -0.1
    assert comparisons[0]["vanilla"]["experiment_id"] == "exp_001_chunk256_ada"
    assert comparisons[0]["llamaindex"]["experiment_id"] == "exp_001_llamaindex"
