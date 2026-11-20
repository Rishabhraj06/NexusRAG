"""Unified high-level orchestrator for NexusRAG multi-agent hybrid pipeline."""

import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from nexusrag.config import NexusSettings
from nexusrag.ingestion import DocumentParser, SemanticChunker, DocumentChunk
from nexusrag.retriever import HybridRetriever
from nexusrag.agents import QueryExpansionAgent, RelevanceEvaluatorAgent, SynthesisAgent, RAGResponse


class NexusEngine:
    """Enterprise RAG engine tying together chunking, hybrid retrieval, and agent evaluation."""

    def __init__(self, settings: Optional[NexusSettings] = None):
        self.settings = settings or NexusSettings()
        self.chunker = SemanticChunker(self.settings.chunking)
        self.retriever = HybridRetriever(self.settings.retriever)
        self.expansion_agent = QueryExpansionAgent()
        self.evaluator_agent = RelevanceEvaluatorAgent(self.settings.agent)
        self.synthesis_agent = SynthesisAgent(self.settings.agent)
        self.all_chunks: List[DocumentChunk] = []

    def ingest_file(self, file_path: Path) -> int:
        """Parse, chunk, and index a single document file."""
        path = Path(file_path)
        text = DocumentParser.parse_file(path)
        chunks = self.chunker.chunk_text(text, source=path.name)
        self.all_chunks.extend(chunks)
        self.retriever.index_documents(self.all_chunks)
        return len(chunks)

    def ingest_text(self, text: str, source_name: str = "raw_input") -> int:
        """Directly chunk and index raw string text."""
        chunks = self.chunker.chunk_text(text, source=source_name)
        self.all_chunks.extend(chunks)
        self.retriever.index_documents(self.all_chunks)
        return len(chunks)

    def query(self, user_query: str) -> Dict[str, Any]:
        """Execute the complete Agentic RAG retrieval, reflection, and synthesis cycle."""
        start_time = time.time()

        # Step 1: Query expansion
        expanded_queries = self.expansion_agent.expand(user_query)
        primary_query = expanded_queries[0]

        # Step 2: Hybrid retrieval (BM25 + Dense Vector with RRF)
        retrieved_chunks = self.retriever.retrieve(primary_query)

        # Step 3: Agentic relevance evaluation (Corrective RAG)
        eval_result = self.evaluator_agent.evaluate(user_query, retrieved_chunks)

        # If insufficient and expansion suggested, retry with expanded query
        if not eval_result.is_sufficient and len(expanded_queries) > 1:
            retry_query = expanded_queries[1]
            retry_chunks = self.retriever.retrieve(retry_query)
            if retry_chunks:
                retrieved_chunks = retry_chunks
                eval_result = self.evaluator_agent.evaluate(retry_query, retrieved_chunks)

        # Step 4: Grounded synthesis
        response: RAGResponse = self.synthesis_agent.synthesize(user_query, retrieved_chunks, eval_result)
        elapsed_ms = (time.time() - start_time) * 1000

        return {
            "query": user_query,
            "answer": response.answer,
            "citations": response.citations,
            "relevance_score": eval_result.relevance_score,
            "is_sufficient": eval_result.is_sufficient,
            "evaluation_reasoning": eval_result.reasoning,
            "model_used": response.model_used,
            "latency_ms": round(elapsed_ms, 2),
            "chunks_count": len(retrieved_chunks),
            "retrieved_chunks": [
                {
                    "source": c.chunk.source,
                    "chunk_id": c.chunk.chunk_id,
                    "content": c.chunk.content,
                    "fused_score": round(c.fused_score, 4),
                    "sparse_score": round(c.sparse_score, 2),
                    "dense_score": round(c.dense_score, 4)
                }
                for c in retrieved_chunks
            ]
        }

    def get_stats(self) -> Dict[str, Any]:
        """Return statistics about currently indexed documents and chunks."""
        unique_docs = len({c.source for c in self.all_chunks})
        total_chars = sum(len(c.content) for c in self.all_chunks)
        return {
            "indexed_documents": unique_docs,
            "total_chunks": len(self.all_chunks),
            "total_characters": total_chars,
            "retriever_type": "Hybrid (Okapi BM25 + Dense Semantic RRF)"
        }
