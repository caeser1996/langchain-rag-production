"""Tests for evaluation metrics."""

import pytest
from src.evaluation.metrics import EvaluationMetrics


@pytest.fixture
def evaluator():
    return EvaluationMetrics(max_history=100)


class TestEvaluationMetrics:
    """Test cases for EvaluationMetrics."""

    @pytest.mark.asyncio
    async def test_evaluate_with_sources(self, evaluator):
        """Test evaluation with sources returns reasonable score."""
        score = await evaluator.evaluate(
            query="What is machine learning?",
            response="Machine learning is a subset of artificial intelligence.",
            sources=[{"text": "ML info", "score": 0.9}]
        )
        assert 0 <= score <= 1

    @pytest.mark.asyncio
    async def test_evaluate_no_sources(self, evaluator):
        """Test evaluation without sources."""
        score = await evaluator.evaluate(
            query="test query",
            response="test response",
            sources=[]
        )
        assert 0 <= score <= 1

    @pytest.mark.asyncio
    async def test_evaluate_empty_response(self, evaluator):
        """Test evaluation with empty response."""
        score = await evaluator.evaluate(
            query="test query",
            response="",
            sources=[]
        )
        assert score < 0.5  # Should be low score

    def test_get_summary_empty(self, evaluator):
        """Test summary with no data."""
        summary = evaluator.get_summary()
        assert summary["total_queries"] == 0
        assert summary["error_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_get_summary_with_data(self, evaluator):
        """Test summary after evaluations."""
        await evaluator.evaluate("q1", "response one with content", [{"score": 0.8}])
        await evaluator.evaluate("q2", "response two with content", [{"score": 0.7}])

        summary = evaluator.get_summary()
        assert summary["total_queries"] == 2
        assert summary["avg_relevance_score"] > 0

    def test_record_error(self, evaluator):
        """Test error recording."""
        evaluator.record_error()
        evaluator.record_error()

        summary = evaluator.get_summary()
        assert summary["total_errors"] == 2

    @pytest.mark.asyncio
    async def test_recent_queries(self, evaluator):
        """Test getting recent queries."""
        await evaluator.evaluate("query1", "response1 content here", [])
        await evaluator.evaluate("query2", "response2 content here", [])

        recent = evaluator.get_recent_queries(limit=5)
        assert len(recent) == 2
        assert recent[0]["query"] == "query2"  # Most recent first

    def test_reset(self, evaluator):
        """Test reset clears all data."""
        evaluator.record_error()
        evaluator.reset()

        summary = evaluator.get_summary()
        assert summary["total_queries"] == 0
        assert summary["total_errors"] == 0

    @pytest.mark.asyncio
    async def test_max_history_limit(self):
        """Test history is trimmed to max size."""
        evaluator = EvaluationMetrics(max_history=5)

        for i in range(10):
            await evaluator.evaluate(f"query{i}", f"response{i} content", [])

        assert len(evaluator._history) == 5
