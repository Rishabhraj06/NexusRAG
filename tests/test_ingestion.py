"""Tests for document parsing and chunking in NexusRAG."""

from nexusrag.ingestion import SemanticChunker, DocumentParser
from nexusrag.config import ChunkingConfig


def test_chunker_creates_valid_chunks():
    config = ChunkingConfig(chunk_size=100, chunk_overlap=20, min_chunk_len=10)
    chunker = SemanticChunker(config)

    text = "Paragraph 1 about artificial intelligence.\n\nParagraph 2 about multi-agent systems and consensus."
    chunks = chunker.chunk_text(text, source="test_doc.md")

    assert len(chunks) >= 1
    assert all(c.doc_id for c in chunks)
    assert all(len(c.content) >= config.min_chunk_len for c in chunks)


def test_chunker_handles_empty_text():
    chunker = SemanticChunker()
    chunks = chunker.chunk_text("", source="empty.txt")
    assert len(chunks) == 0
