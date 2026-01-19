"""Document processor for PDF, DOCX, and medical records."""

from typing import List, Dict, Any, Optional
import io
import os


class DocumentProcessor:
    """Process various document formats for RAG ingestion."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    async def process_file(
        self,
        content: bytes,
        filename: str
    ) -> List[Dict[str, Any]]:
        """Process a file and return chunks with metadata."""
        ext = os.path.splitext(filename)[1].lower()

        if ext == ".pdf":
            text = await self._extract_pdf(content)
        elif ext in [".docx", ".doc"]:
            text = await self._extract_docx(content)
        elif ext == ".txt":
            text = content.decode("utf-8")
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        chunks = self._chunk_text(text)

        return [
            {
                "text": chunk,
                "metadata": {
                    "source": filename,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            }
            for i, chunk in enumerate(chunks)
        ]

    async def _extract_pdf(self, content: bytes) -> str:
        """Extract text from PDF."""
        try:
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(content))
            text_parts = []

            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)

            return "\n\n".join(text_parts)
        except ImportError:
            raise RuntimeError("pypdf is required for PDF processing")

    async def _extract_docx(self, content: bytes) -> str:
        """Extract text from DOCX."""
        try:
            from docx import Document

            doc = Document(io.BytesIO(content))
            text_parts = []

            for para in doc.paragraphs:
                if para.text.strip():
                    text_parts.append(para.text)

            return "\n\n".join(text_parts)
        except ImportError:
            raise RuntimeError("python-docx is required for DOCX processing")

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        if not text:
            return []

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                for sep in [". ", ".\n", "! ", "? "]:
                    last_sep = text[start:end].rfind(sep)
                    if last_sep != -1:
                        end = start + last_sep + len(sep)
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start position with overlap
            start = end - self.chunk_overlap
            if start < 0:
                start = end

        return chunks

    async def process_medical_record(
        self,
        content: bytes,
        filename: str,
        patient_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Process medical records with additional metadata extraction."""
        chunks = await self.process_file(content, filename)

        # Add medical-specific metadata
        for chunk in chunks:
            chunk["metadata"]["document_type"] = "medical_record"
            if patient_id:
                chunk["metadata"]["patient_id"] = patient_id

            # Extract potential medical entities (simplified)
            text_lower = chunk["text"].lower()
            if any(term in text_lower for term in ["diagnosis", "dx", "assessment"]):
                chunk["metadata"]["contains_diagnosis"] = True
            if any(term in text_lower for term in ["medication", "prescription", "rx"]):
                chunk["metadata"]["contains_medication"] = True
            if any(term in text_lower for term in ["lab", "test", "result"]):
                chunk["metadata"]["contains_lab_results"] = True

        return chunks
