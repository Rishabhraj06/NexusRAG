"""End-to-end tests for the complete NexusRAG pipeline and agentic reflection."""

from nexusrag.pipeline import NexusEngine


def test_end_to_end_query():
    engine = NexusEngine()
    sample_text = (
        "PostgreSQL is a powerful, open-source object-relational database system.\n\n"
        "Redis is an in-memory data structure store used as a database, cache, and message broker."
    )
    engine.ingest_text(sample_text, source_name="databases.txt")

    res = engine.query("What is Redis used for?")

    assert res["query"] == "What is Redis used for?"
    assert len(res["answer"]) > 10
    assert res["chunks_count"] > 0
    assert "Redis" in res["answer"]
    assert res["latency_ms"] >= 0.0


def test_empty_knowledge_base_query():
    engine = NexusEngine()
    res = engine.query("Hello test")
    assert res["chunks_count"] == 0
    assert "No relevant" in res["answer"]
