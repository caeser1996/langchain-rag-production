"""RAG Orchestrator using LangGraph for multi-agent coordination."""

from typing import Optional, Dict, Any, List, TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
import hashlib


class RAGState(TypedDict):
    """State for RAG pipeline."""
    query: str
    rewritten_query: str
    retrieved_docs: List[dict]
    response: str
    sources: List[dict]
    evaluation_score: float
    cached: bool
    error: Optional[str]


class RAGOrchestrator:
    """Orchestrates the RAG pipeline using LangGraph."""

    def __init__(self, vector_store, cache, evaluator):
        self.vector_store = vector_store
        self.cache = cache
        self.evaluator = evaluator
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(RAGState)

        # Add nodes
        workflow.add_node("rewrite_query", self._rewrite_query)
        workflow.add_node("retrieve", self._retrieve_documents)
        workflow.add_node("generate", self._generate_response)
        workflow.add_node("evaluate", self._evaluate_response)

        # Add edges
        workflow.set_entry_point("rewrite_query")
        workflow.add_edge("rewrite_query", "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "evaluate")
        workflow.add_edge("evaluate", END)

        return workflow.compile()

    async def _rewrite_query(self, state: RAGState) -> RAGState:
        """Rewrite query for better retrieval."""
        # Simple query rewriting - expand abbreviations and normalize
        query = state["query"].strip()

        # Basic query optimization
        rewritten = query.lower()

        # Expand common medical abbreviations if present
        abbreviations = {
            "bp": "blood pressure",
            "hr": "heart rate",
            "temp": "temperature",
            "rx": "prescription",
            "dx": "diagnosis",
        }

        for abbr, expansion in abbreviations.items():
            if abbr in rewritten.split():
                rewritten = rewritten.replace(abbr, expansion)

        state["rewritten_query"] = rewritten
        return state

    async def _retrieve_documents(self, state: RAGState) -> RAGState:
        """Retrieve relevant documents from vector store."""
        try:
            docs = await self.vector_store.search(
                query=state["rewritten_query"],
                top_k=5
            )
            state["retrieved_docs"] = docs
            state["sources"] = [
                {"text": doc["text"][:200], "score": doc["score"]}
                for doc in docs
            ]
        except Exception as e:
            state["error"] = f"Retrieval error: {str(e)}"
            state["retrieved_docs"] = []
            state["sources"] = []

        return state

    async def _generate_response(self, state: RAGState) -> RAGState:
        """Generate response using retrieved context."""
        if state.get("error"):
            state["response"] = "Unable to generate response due to retrieval error."
            return state

        # Build context from retrieved documents
        context = "\n\n".join([
            doc["text"] for doc in state["retrieved_docs"]
        ])

        if not context:
            state["response"] = "No relevant documents found for your query."
            return state

        # Generate response (placeholder - would use LLM in production)
        state["response"] = f"Based on the retrieved documents: {context[:500]}..."

        return state

    async def _evaluate_response(self, state: RAGState) -> RAGState:
        """Evaluate the quality of the response."""
        try:
            score = await self.evaluator.evaluate(
                query=state["query"],
                response=state["response"],
                sources=state["retrieved_docs"]
            )
            state["evaluation_score"] = score
        except Exception:
            state["evaluation_score"] = 0.0

        return state

    def _get_cache_key(self, query: str, top_k: int, filters: Optional[dict]) -> str:
        """Generate cache key for query."""
        key_data = f"{query}:{top_k}:{str(filters)}"
        return hashlib.md5(key_data.encode()).hexdigest()

    async def process_query(
        self,
        query: str,
        top_k: int = 5,
        use_cache: bool = True,
        filters: Optional[dict] = None
    ) -> Dict[str, Any]:
        """Process a RAG query through the pipeline."""

        # Check cache first
        if use_cache:
            cache_key = self._get_cache_key(query, top_k, filters)
            cached_result = await self.cache.get(cache_key)
            if cached_result:
                cached_result["cached"] = True
                return cached_result

        # Initialize state
        initial_state: RAGState = {
            "query": query,
            "rewritten_query": "",
            "retrieved_docs": [],
            "response": "",
            "sources": [],
            "evaluation_score": 0.0,
            "cached": False,
            "error": None
        }

        # Run the graph
        final_state = await self.graph.ainvoke(initial_state)

        result = {
            "answer": final_state["response"],
            "sources": final_state["sources"],
            "cached": False,
            "evaluation_score": final_state["evaluation_score"]
        }

        # Cache the result
        if use_cache:
            await self.cache.set(cache_key, result)

        return result
