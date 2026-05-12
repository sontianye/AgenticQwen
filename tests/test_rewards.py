#!/usr/bin/env python3
"""Unit tests for math and agentic reward functions."""

from __future__ import annotations

import pytest
from agentic_qwen.training.rewards.math_reward import compute_math_score
from agentic_qwen.training.rewards.agentic_reward import compute_agentic_score


# ---------------------------------------------------------------------------
# Math reward
# ---------------------------------------------------------------------------

class TestMathReward:

    def test_correct_answer_no_tools(self):
        assert compute_math_score("", "Let me think.\n<answer>42</answer>", "42") == 1.0

    def test_wrong_answer(self):
        assert compute_math_score("", "<answer>99</answer>", "42") == 0.0

    def test_empty_solution(self):
        assert compute_math_score("", "", "42") == 0.0

    def test_correct_with_tool_call_scores_lower(self):
        solution = '<tool_call>{"name": "calc"}</tool_call>\n<answer>42</answer>'
        assert compute_math_score("", solution, "42") == 0.9

    def test_case_insensitive_match(self):
        assert compute_math_score("", "<answer>Paris</answer>", "paris") == 1.0

    def test_fallback_to_last_line(self):
        assert compute_math_score("", "Step 1: …\n42", "42") == 1.0

    def test_score_is_float(self):
        score = compute_math_score("", "<answer>7</answer>", "7")
        assert isinstance(score, float)


# ---------------------------------------------------------------------------
# Agentic reward
# ---------------------------------------------------------------------------

class TestAgenticReward:

    def test_completion_marker_raises_score(self):
        solution = "I have completed the booking.\n###STOP"
        score = compute_agentic_score("", solution, "")
        assert score >= 0.4

    def test_empty_solution_is_near_zero(self):
        assert compute_agentic_score("", "", "") < 0.2

    def test_tool_calls_contribute_to_score(self):
        solution = '<tool_call>{"name": "search"}</tool_call>\nResults found.\n###STOP'
        score = compute_agentic_score("", solution, "", {"rubrics": ["search for results"]})
        assert score >= 0.4

    def test_score_always_in_unit_interval(self):
        for solution in ["", "hello", "###STOP", '<tool_call/>' * 20, "x" * 10_000]:
            score = compute_agentic_score("", solution, {})  # type: ignore[arg-type]
            assert 0.0 <= score <= 1.0, f"Score {score} out of [0, 1] for input of length {len(str(solution))}"

    def test_transfer_marker_also_counts_as_completion(self):
        solution = "Escalating to human.\n###TRANSFER_TO_HUMAN"
        score = compute_agentic_score("", solution, "")
        assert score >= 0.4

    def test_rubrics_influence_tool_weight(self):
        # More rubrics → proportionally lower tool-call score when only one call is made
        one_rubric = compute_agentic_score("", '<tool_call/>\n###STOP', "", {"rubrics": ["r1"]})
        five_rubrics = compute_agentic_score(
            "", '<tool_call/>\n###STOP', "", {"rubrics": [f"r{i}" for i in range(5)]}
        )
        assert one_rubric >= five_rubrics
