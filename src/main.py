"""Main FastAPI application for RAG pipeline."""

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional, List
import os

from src.agents.orchestrator import RAGOrchestrator
from src.ingestion.document_processor import DocumentProcessor
from src.retrieval.vector_store import VectorStoreManager
from src.cache.redis_cache import RedisCache
from src.evaluation.metrics import EvaluationMetrics

app = FastAPI(
    title="RAG Production Pipeline",
    description="Production-ready Retrieval-Augmented Generation API",
    version="0.1.0"
)

# Initialize components
orchestrator = None
doc_processor = None
vector_store = None
cache = None
evaluator = None


class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    use_cache: Optional[bool] = True
    filters: Optional[dict] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    cached: bool
    evaluation_score: Optional[float] = None


class IngestRequest(BaseModel):
    text: str
    metadata: Optional[dict] = None


@app.on_event("startup")
async def startup_event():
    """Initialize all components on startup."""
    global orchestrator, doc_processor, vector_store, cache, evaluator

    cache = RedisCache()
    vector_store = VectorStoreManager()
    doc_processor = DocumentProcessor()
    evaluator = EvaluationMetrics()
    orchestrator = RAGOrchestrator(
        vector_store=vector_store,
        cache=cache,
        evaluator=evaluator
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "0.1.0"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Process a RAG query."""
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        result = await orchestrator.process_query(
            query=request.query,
            top_k=request.top_k,
            use_cache=request.use_cache,
            filters=request.filters
        )
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest")
async def ingest_text(request: IngestRequest):
    """Ingest text directly into the vector store."""
    if vector_store is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        doc_id = await vector_store.add_document(
            text=request.text,
            metadata=request.metadata
        )
        return {"status": "success", "document_id": doc_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    """Ingest a document file (PDF, DOCX)."""
    if doc_processor is None or vector_store is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        content = await file.read()
        filename = file.filename

        # Process the document
        chunks = await doc_processor.process_file(content, filename)

        # Add to vector store
        doc_ids = []
        for chunk in chunks:
            doc_id = await vector_store.add_document(
                text=chunk["text"],
                metadata=chunk["metadata"]
            )
            doc_ids.append(doc_id)

        return {
            "status": "success",
            "filename": filename,
            "chunks_processed": len(doc_ids),
            "document_ids": doc_ids
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """Get evaluation metrics."""
    if evaluator is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    return evaluator.get_summary()


@app.delete("/cache")
async def clear_cache():
    """Clear the Redis cache."""
    if cache is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    await cache.clear()
    return {"status": "success", "message": "Cache cleared"}
