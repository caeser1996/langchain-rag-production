"""Evaluation metrics for RAG pipeline."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import statistics


@dataclass
class QueryMetrics:
    """Metrics for a single query."""
    query: str
    response_length: int
    num_sources: int
    relevance_score: float
    latency_ms: float
    timestamp: datetime = field(default_factory=datetime.now)


class EvaluationMetrics:
    """Track and compute evaluation metrics for RAG responses."""

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self._history: List[QueryMetrics] = []
        self._total_queries = 0
        self._total_errors = 0

    async def evaluate(
        self,
        query: str,
        response: str,
        sources: List[Dict[str, Any]],
        latency_ms: Optional[float] = None
    ) -> float:
        """Evaluate a RAG response and return a relevance score."""

        # Compute relevance score based on multiple factors
        score = 0.0

        # Factor 1: Source coverage (do we have sources?)
        if sources:
            source_score = min(len(sources) / 3, 1.0) * 0.3
            score += source_score

        # Factor 2: Response quality (not empty, reasonable length)
        if response and len(response) > 50:
            length_score = min(len(response) / 500, 1.0) * 0.3
            score += length_score

        # Factor 3: Query-response overlap (simple keyword matching)
        query_terms = set(query.lower().split())
        response_terms = set(response.lower().split())
        if query_terms:
            overlap = len(query_terms & response_terms) / len(query_terms)
            score += overlap * 0.2

        # Factor 4: Source quality (based on retrieval scores)
        if sources:
            avg_source_score = sum(s.get("score", 0) for s in sources) / len(sources)
            score += avg_source_score * 0.2

        # Record metrics
        metrics = QueryMetrics(
            query=query,
            response_length=len(response),
            num_sources=len(sources),
            relevance_score=score,
            latency_ms=latency_ms or 0
        )
        self._record_metrics(metrics)

        return score

    def _record_metrics(self, metrics: QueryMetrics):
        """Record metrics for history tracking."""
        self._history.append(metrics)
        self._total_queries += 1

        # Trim history if needed
        if len(self._history) > self.max_history:
            self._history = self._history[-self.max_history:]

    def record_error(self):
        """Record an error occurrence."""
        self._total_errors += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self._history:
            return {
                "total_queries": self._total_queries,
                "total_errors": self._total_errors,
                "error_rate": 0.0,
                "avg_relevance_score": 0.0,
                "avg_response_length": 0,
                "avg_sources_per_query": 0,
                "avg_latency_ms": 0
            }

        relevance_scores = [m.relevance_score for m in self._history]
        response_lengths = [m.response_length for m in self._history]
        source_counts = [m.num_sources for m in self._history]
        latencies = [m.latency_ms for m in self._history if m.latency_ms > 0]

        error_rate = (
            self._total_errors / self._total_queries
            if self._total_queries > 0 else 0.0
        )

        return {
            "total_queries": self._total_queries,
            "total_errors": self._total_errors,
            "error_rate": round(error_rate, 4),
            "avg_relevance_score": round(statistics.mean(relevance_scores), 4),
            "min_relevance_score": round(min(relevance_scores), 4),
            "max_relevance_score": round(max(relevance_scores), 4),
            "avg_response_length": round(statistics.mean(response_lengths)),
            "avg_sources_per_query": round(statistics.mean(source_counts), 2),
            "avg_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0,
            "p95_latency_ms": round(
                sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0, 2
            )
        }

    def get_recent_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent query metrics."""
        recent = self._history[-limit:]
        return [
            {
                "query": m.query[:100],
                "relevance_score": round(m.relevance_score, 4),
                "response_length": m.response_length,
                "num_sources": m.num_sources,
                "latency_ms": round(m.latency_ms, 2),
                "timestamp": m.timestamp.isoformat()
            }
            for m in reversed(recent)
        ]

    def reset(self):
        """Reset all metrics."""
        self._history = []
        self._total_queries = 0
        self._total_errors = 0
