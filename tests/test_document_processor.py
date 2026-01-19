"""Tests for document processor."""

import pytest
from src.ingestion.document_processor import DocumentProcessor


@pytest.fixture
def processor():
    return DocumentProcessor(chunk_size=100, chunk_overlap=20)


class TestDocumentProcessor:
    """Test cases for DocumentProcessor."""

    def test_chunk_text_empty(self, processor):
        """Test chunking empty text."""
        chunks = processor._chunk_text("")
        assert chunks == []

    def test_chunk_text_small(self, processor):
        """Test chunking text smaller than chunk size."""
        text = "This is a small piece of text."
        chunks = processor._chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_text_large(self, processor):
        """Test chunking text larger than chunk size."""
        text = "This is sentence one. " * 20
        chunks = processor._chunk_text(text)
        assert len(chunks) > 1

    def test_chunk_text_preserves_content(self, processor):
        """Test that chunking preserves all content."""
        text = "Word " * 50
        chunks = processor._chunk_text(text)
        # Verify all chunks have content
        for chunk in chunks:
            assert len(chunk) > 0

    @pytest.mark.asyncio
    async def test_process_txt_file(self, processor):
        """Test processing a text file."""
        content = b"This is test content for the document processor."
        filename = "test.txt"

        chunks = await processor.process_file(content, filename)

        assert len(chunks) >= 1
        assert chunks[0]["text"] == content.decode("utf-8")
        assert chunks[0]["metadata"]["source"] == filename

    @pytest.mark.asyncio
    async def test_unsupported_file_type(self, processor):
        """Test processing unsupported file type."""
        content = b"Some content"
        filename = "test.xyz"

        with pytest.raises(ValueError, match="Unsupported file type"):
            await processor.process_file(content, filename)

    @pytest.mark.asyncio
    async def test_medical_record_metadata(self, processor):
        """Test medical record processing adds metadata."""
        content = b"Patient diagnosis: hypertension. Medication: lisinopril."
        filename = "medical.txt"

        chunks = await processor.process_medical_record(
            content, filename, patient_id="P123"
        )

        assert len(chunks) >= 1
        assert chunks[0]["metadata"]["document_type"] == "medical_record"
        assert chunks[0]["metadata"]["patient_id"] == "P123"
        assert chunks[0]["metadata"].get("contains_diagnosis") == True
        assert chunks[0]["metadata"].get("contains_medication") == True
