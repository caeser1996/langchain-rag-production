"""Vector store manager for Pinecone and ChromaDB."""

from typing import List, Dict, Any, Optional
import logging
import os
import uuid

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manage vector store operations with Pinecone and ChromaDB support."""

    def __init__(
        self,
        backend: str = "chroma",
        embedding_model: str = "text-embedding-3-small"
    ):
        self.backend = backend
        self.embedding_model = embedding_model
        self._client = None
        self._embeddings = None

    async def _get_embeddings(self):
        """Get or create embeddings model."""
        if self._embeddings is None:
            try:
                from langchain_openai import OpenAIEmbeddings
                self._embeddings = OpenAIEmbeddings(model=self.embedding_model)
            except ImportError:
                raise RuntimeError("langchain-openai is required for embeddings")
        return self._embeddings

    async def _get_client(self):
        """Get or create vector store client."""
        if self._client is not None:
            return self._client

        if self.backend == "pinecone":
            self._client = await self._init_pinecone()
        else:
            self._client = await self._init_chroma()

        return self._client

    async def _init_pinecone(self):
        """Initialize Pinecone client."""
        try:
            from pinecone import Pinecone

            api_key = os.getenv("PINECONE_API_KEY")
            if not api_key:
                raise ValueError("PINECONE_API_KEY environment variable not set")

            pc = Pinecone(api_key=api_key)
            index_name = os.getenv("PINECONE_INDEX_NAME", "rag-production")

            return pc.Index(index_name)
        except ImportError:
            raise RuntimeError("pinecone-client is required for Pinecone backend")

    async def _init_chroma(self):
        """Initialize ChromaDB client."""
        try:
            import chromadb
            from chromadb.config import Settings

            persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")

            client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=persist_dir,
                anonymized_telemetry=False
            ))

            # Get or create collection
            collection = client.get_or_create_collection(
                name="rag_documents",
                metadata={"hnsw:space": "cosine"}
            )

            return collection
        except ImportError:
            raise RuntimeError("chromadb is required for ChromaDB backend")

    async def add_document(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a document to the vector store."""
        doc_id = str(uuid.uuid4())
        embeddings = await self._get_embeddings()
        client = await self._get_client()

        # Generate embedding
        embedding = await embeddings.aembed_query(text)

        if self.backend == "pinecone":
            client.upsert(vectors=[{
                "id": doc_id,
                "values": embedding,
                "metadata": {**(metadata or {}), "text": text}
            }])
        else:
            client.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[metadata or {}]
            )

        return doc_id

    async def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        embeddings = await self._get_embeddings()
        client = await self._get_client()

        # Generate query embedding
        query_embedding = await embeddings.aembed_query(query)

        if self.backend == "pinecone":
            results = client.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filters
            )

            return [
                {
                    "id": match["id"],
                    "text": match["metadata"].get("text", ""),
                    "score": match["score"],
                    "metadata": match["metadata"]
                }
                for match in results["matches"]
            ]
        else:
            results = client.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filters
            )

            documents = []
            if results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    documents.append({
                        "id": doc_id,
                        "text": results["documents"][0][i] if results["documents"] else "",
                        "score": 1 - results["distances"][0][i] if results["distances"] else 0,
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {}
                    })

            return documents

    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the vector store."""
        client = await self._get_client()

        try:
            client.delete(ids=[doc_id])
            logger.debug(f"Deleted document {doc_id} from {self.backend}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        client = await self._get_client()

        if self.backend == "pinecone":
            stats = client.describe_index_stats()
            return {
                "backend": "pinecone",
                "total_vectors": stats.get("total_vector_count", 0),
                "dimension": stats.get("dimension", 0)
            }
        else:
            return {
                "backend": "chroma",
                "total_documents": client.count()
            }
