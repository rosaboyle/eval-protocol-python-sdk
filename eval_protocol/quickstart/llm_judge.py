"""
Default LLM judge for Eval Protocol. Inspired by Arena-Hard-Auto.
"""

import os
from typing import List, Dict, Any, Optional
from tqdm import tqdm

import pytest

from eval_protocol.models import EvaluateResult, EvaluationRow, MetricResult
from eval_protocol.pytest import evaluation_test
from eval_protocol.pytest.default_single_turn_rollout_process import SingleTurnRolloutProcessor
from eval_protocol.quickstart.utils import (
    split_multi_turn_rows,
    JUDGE_CONFIGS,
    fetch_langfuse_traces_as_evaluation_rows,
    calculate_bootstrap_scores,
    push_scores_to_langfuse,
    run_judgment,
)

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Skip in CI")
@pytest.mark.asyncio
@evaluation_test(
    input_rows=[fetch_langfuse_traces_as_evaluation_rows()],
    completion_params=[
        {
            "model": "fireworks_ai/accounts/fireworks/models/qwen3-235b-a22b-instruct-2507",
        },
        {
            "max_tokens": 131000,
            "extra_body": {"reasoning_effort": "low"},
            "model": "fireworks_ai/accounts/fireworks/models/gpt-oss-120b",
        },
    ],
    rollout_processor=SingleTurnRolloutProcessor(),
    preprocess_fn=split_multi_turn_rows,
    mode="all",
)
async def test_llm_judge(rows: list[EvaluationRow]) -> list[EvaluationRow]:
    """
    Simplified LLM Judge for Arena-Hard-Auto pairwise comparisons.

    Each row contains:
    - messages[:-1]: Question/prompt (conversation context)
    - messages[-1]: Model B's answer (comparison model response)
    - ground_truth: Model A's answer (original assistant response)
    """

    judge_name = "gemini-2.5-pro"  # Edit to which judge you'd like to use. Configs are in utils.py.

    if not rows:
        print("❌ No evaluation rows provided")
        return rows

    print(f"🔄 Processing {len(rows)} evaluation rows for LLM judging...")

    model_name = rows[0].input_metadata.completion_params.get("model", "unknown_model")

    judgments = []
    max_concurrency = JUDGE_CONFIGS[judge_name]["max_concurrency"]

    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures = [executor.submit(run_judgment, row, model_name, judge_name) for row in rows]

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Generating judgments"):
            result = future.result()
            if result and result["games"][0] and result["games"][1]:
                judgments.append(result)

    if not judgments:
        print("❌ No valid judgments generated")
        return rows

    print(f"✅ Generated {len(judgments)} valid judgments")

    # Calculate bootstrap scores
    mean_score, lower_score, upper_score = calculate_bootstrap_scores(judgments)

    if mean_score == 0.0:
        print("❌ No valid scores extracted")
        return rows

    # Print leaderboard
    print("\n##### LLM Judge Results (90th percentile CI) #####")

    clean_model_name = model_name.split("/")[-1]  # Clean model name

    print(f"{clean_model_name}: {mean_score:.1%} (CI: {lower_score:.1%} - {upper_score:.1%})")
    print("original: 50.0% (CI: 50.0% - 50.0%)")

    for row in rows:
        if row.evaluation_result:
            row.evaluation_result.score = mean_score
            row.evaluation_result.standard_error = (upper_score - lower_score) / (
                2 * 1.645
            )  # Standard error approximation from 90% CI

    # Optional, push scores back to Langfuse. Note that one score per model will be pushed back onto same trace.
    push_scores_to_langfuse(rows, model_name, mean_score)

    return rows
