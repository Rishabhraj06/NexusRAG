"""Agentic reasoning: Query Expansion, Corrective Evaluation, and Synthesis."""

import os
import re
from dataclasses import dataclass
from typing import List, Optional

from nexusrag.config import AgentConfig
from nexusrag.retriever import ScoredChunk


@dataclass
class EvaluationResult:
    is_sufficient: bool
    relevance_score: float
    reasoning: str
    suggested_queries: List[str]


@dataclass
class RAGResponse:
    query: str
    answer: str
    citations: List[str]
    retrieved_chunks: List[ScoredChunk]
    evaluation: EvaluationResult
    model_used: str


class QueryExpansionAgent:
    """Expands short or colloquial queries into targeted technical keywords."""

    def expand(self, query: str) -> List[str]:
        words = re.findall(r"\b\w+\b", query.lower())
        stopwords = {"what", "is", "the", "how", "to", "in", "and", "a", "an", "does", "do", "explain"}
        keywords = [w for w in words if w not in stopwords]

        expansions = [query]
        if keywords:
            expansions.append(" ".join(keywords))
        return expansions


class RelevanceEvaluatorAgent:
    """Evaluates whether retrieved passages are relevant to the user's question (CRAG)."""

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()

    def evaluate(self, query: str, chunks: List[ScoredChunk]) -> EvaluationResult:
        if not chunks:
            return EvaluationResult(
                is_sufficient=False,
                relevance_score=0.0,
                reasoning="No candidate chunks retrieved.",
                suggested_queries=[query]
            )

        q_terms = set(re.findall(r"\b\w{3,}\b", query.lower()))
        if not q_terms:
            return EvaluationResult(True, 1.0, "Query terms matched.", [])

        matched_terms = set()
        for scored in chunks:
            text = scored.chunk.content.lower()
            for term in q_terms:
                if term in text:
                    matched_terms.add(term)

        overlap_ratio = len(matched_terms) / len(q_terms)
        is_sufficient = overlap_ratio >= self.config.relevance_threshold

        reasoning = f"Matched {len(matched_terms)}/{len(q_terms)} query terms ({overlap_ratio*100:.0f}% coverage)."
        suggestions = []
        if not is_sufficient:
            missing = q_terms - matched_terms
            suggestions = [f"{query} {' '.join(missing)}"]

        return EvaluationResult(
            is_sufficient=is_sufficient,
            relevance_score=overlap_ratio,
            reasoning=reasoning,
            suggested_queries=suggestions
        )


class SynthesisAgent:
    """Synthesizes accurate answers grounded strictly in retrieved context."""

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()

    def synthesize(self, query: str, chunks: List[ScoredChunk], evaluation: EvaluationResult) -> RAGResponse:
        citations = [f"{c.chunk.source} (chunk {c.chunk.chunk_index})" for c in chunks]

        # Check for Gemini API key
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key and chunks:
            try:
                answer = self._synthesize_with_gemini(query, chunks, api_key)
                return RAGResponse(
                    query=query,
                    answer=answer,
                    citations=citations,
                    retrieved_chunks=chunks,
                    evaluation=evaluation,
                    model_used="gemini-2.5-flash"
                )
            except Exception:
                pass

        # Intelligent Extractive Synthesis (Offline / Fallback)
        answer = self._synthesize_extractive(query, chunks, evaluation)
        return RAGResponse(
            query=query,
            answer=answer,
            citations=citations,
            retrieved_chunks=chunks,
            evaluation=evaluation,
            model_used="extractive-synthesizer"
        )

    def _synthesize_with_gemini(self, query: str, chunks: List[ScoredChunk], api_key: str) -> str:
        from google import genai
        client = genai.Client(api_key=api_key)

        context_blocks = []
        for idx, scored in enumerate(chunks, 1):
            context_blocks.append(f"[Document {idx} - Source: {scored.chunk.source}]\n{scored.chunk.content}")

        context_str = "\n\n".join(context_blocks)
        prompt = (
            "You are NexusRAG, an enterprise AI research assistant.\n"
            "Answer the user query strictly using the verified context below.\n"
            "Cite sources where relevant. If the context does not contain enough info, state clearly.\n\n"
            f"=== CONTEXT ===\n{context_str}\n\n"
            f"=== USER QUERY ===\n{query}\n\n"
            "Answer:"
        )

        response = client.models.generate_content(
            model=self.config.model_name,
            contents=prompt
        )
        return response.text.strip()

    def _synthesize_extractive(self, query: str, chunks: List[ScoredChunk], evaluation: EvaluationResult) -> str:
        if not chunks:
            return "No relevant documentation found to answer your inquiry."

        top_passages = [c.chunk.content.strip() for c in chunks[:2]]
        bullets = "\n\n".join(f"> \"{p}\"" for p in top_passages)

        confidence = "High" if evaluation.is_sufficient else "Partial"
        return (
            f"**Synthesis (Confidence: {confidence}):**\n\n"
            f"Based on the indexed knowledge base:\n\n{bullets}\n\n"
            f"*Relevance Assessment:* {evaluation.reasoning}"
        )
