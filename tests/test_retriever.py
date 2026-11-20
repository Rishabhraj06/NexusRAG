"""Tests for BM25, Dense Vector search, and Reciprocal Rank Fusion."""

from nexusrag.ingestion import SemanticChunker
from nexusrag.retriever import HybridRetriever, BM25Retriever


def test_bm25_retrieval_ranking():
    chunker = SemanticChunker()
    doc1 = "Python is an interpreted, high-level, general-purpose programming language."
    doc2 = "Raft is a distributed consensus algorithm designed for log replication and safety."

    chunks = chunker.chunk_text(doc1 + "\n\n" + doc2, source="docs.txt")
    bm25 = BM25Retriever()
    bm25.index(chunks)

    results = bm25.search("consensus algorithm")
    assert len(results) > 0
    top_chunk, score = results[0]
    assert "consensus" in top_chunk.content.lower()


def test_hybrid_rrf_retrieval():
    chunker = SemanticChunker()
    doc1 = "Microservices architecture uses lightweight protocols like HTTP and gRPC."
    doc2 = "Monolithic applications bundle all services into a single unified binary."

    chunks = chunker.chunk_text(doc1 + "\n\n" + doc2, source="arch.txt")
    retriever = HybridRetriever()
    retriever.index_documents(chunks)

    scored_chunks = retriever.retrieve("gRPC microservices")
    assert len(scored_chunks) > 0
    assert scored_chunks[0].fused_score > 0.0
    assert "microservices" in scored_chunks[0].chunk.content.lower()
