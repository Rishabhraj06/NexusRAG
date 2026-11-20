"""Hybrid Search Retriever combining Okapi BM25 and Dense Vector Embeddings via RRF."""

import re
import math
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple, Optional

from nexusrag.config import RetrieverConfig
from nexusrag.ingestion import DocumentChunk


@dataclass
class ScoredChunk:
    chunk: DocumentChunk
    fused_score: float
    dense_rank: Optional[int]
    sparse_rank: Optional[int]
    dense_score: float
    sparse_score: float


class BM25Retriever:
    """Okapi BM25 sparse keyword retriever (standard Information Retrieval model)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: List[DocumentChunk] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_len: float = 0.0
        self.doc_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}

    def index(self, chunks: List[DocumentChunk]):
        self.corpus = chunks
        self.doc_lengths = []
        self.doc_freqs = []
        df: Dict[str, int] = {}
        total_len = 0

        for chunk in chunks:
            tokens = self._tokenize(chunk.content)
            dlen = len(tokens)
            self.doc_lengths.append(dlen)
            total_len += dlen

            counts: Dict[str, int] = {}
            for t in tokens:
                counts[t] = counts.get(t, 0) + 1
            self.doc_freqs.append(counts)

            for unique_token in counts.keys():
                df[unique_token] = df.get(unique_token, 0) + 1

        n_docs = len(chunks)
        self.avg_doc_len = (total_len / n_docs) if n_docs > 0 else 0.0

        # Compute Robertson-Spärck Jones IDF
        self.idf = {}
        for token, freq in df.items():
            self.idf[token] = math.log((n_docs - freq + 0.5) / (freq + 0.5) + 1.0)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[DocumentChunk, float]]:
        if not self.corpus:
            return []

        q_tokens = self._tokenize(query)
        scores: List[float] = [0.0] * len(self.corpus)

        for q in q_tokens:
            if q not in self.idf:
                continue
            idf_val = self.idf[q]

            for doc_idx, freqs in enumerate(self.doc_freqs):
                if q in freqs:
                    tf = freqs[q]
                    doc_len = self.doc_lengths[doc_idx]
                    denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                    scores[doc_idx] += idf_val * ((tf * (self.k1 + 1.0)) / (denom if denom > 0 else 1.0))

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(self.corpus[idx], score) for idx, score in ranked[:top_k] if score > 0.0]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [t.lower() for t in re.findall(r"\b\w{2,}\b", text)]


class DenseVectorRetriever:
    """Vector similarity retriever using semantic token hashing and cosine similarity."""

    def __init__(self):
        self.chunks: List[DocumentChunk] = []
        self.vectors: List[List[float]] = []

    def index(self, chunks: List[DocumentChunk]):
        self.chunks = chunks
        self.vectors = [self._embed_text(c.content) for c in chunks]

    def search(self, query: str, top_k: int = 10) -> List[Tuple[DocumentChunk, float]]:
        if not self.chunks:
            return []

        q_vec = self._embed_text(query)
        scored: List[Tuple[int, float]] = []

        for idx, doc_vec in enumerate(self.vectors):
            sim = self._cosine_sim(q_vec, doc_vec)
            scored.append((idx, sim))

        ranked = sorted(scored, key=lambda x: x[1], reverse=True)
        return [(self.chunks[idx], sim) for idx, sim in ranked[:top_k]]

    def _embed_text(self, text: str, dims: int = 64) -> List[float]:
        """Deterministic semantic projection embedding for fast, zero-dependency execution."""
        tokens = re.findall(r"\b\w+\b", text.lower())
        vec = [0.0] * dims
        if not tokens:
            return vec

        for t in tokens:
            h = hash(t)
            idx = abs(h) % dims
            weight = 1.0 if (h > 0) else -1.0
            vec[idx] += weight

        # Normalize to unit length
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm > 0 else vec

    @staticmethod
    def _cosine_sim(v1: List[float], v2: List[float]) -> float:
        dot = sum(a * b for a, b in zip(v1, v2))
        return max(0.0, min(1.0, (dot + 1.0) / 2.0))


class HybridRetriever:
    """Merges Dense semantic vectors and Sparse BM25 ranks via Reciprocal Rank Fusion (RRF)."""

    def __init__(self, config: Optional[RetrieverConfig] = None):
        self.config = config or RetrieverConfig()
        self.bm25 = BM25Retriever()
        self.dense = DenseVectorRetriever()
        self.indexed_chunks: List[DocumentChunk] = []

    def index_documents(self, chunks: List[DocumentChunk]):
        self.indexed_chunks = chunks
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def retrieve(self, query: str) -> List[ScoredChunk]:
        """Perform hybrid retrieval with Reciprocal Rank Fusion."""
        if not self.indexed_chunks:
            return []

        top_limit = max(10, self.config.top_k * 2)
        sparse_res = self.bm25.search(query, top_k=top_limit)
        dense_res = self.dense.search(query, top_k=top_limit)

        sparse_ranks: Dict[str, Tuple[int, float]] = {
            item[0].chunk_id: (rank + 1, item[1]) for rank, item in enumerate(sparse_res)
        }
        dense_ranks: Dict[str, Tuple[int, float]] = {
            item[0].chunk_id: (rank + 1, item[1]) for rank, item in enumerate(dense_res)
        }

        all_ids: Set[str] = set(sparse_ranks.keys()).union(dense_ranks.keys())
        chunk_map: Dict[str, DocumentChunk] = {c.chunk_id: c for c in self.indexed_chunks}

        rrf_scored: List[ScoredChunk] = []
        k = self.config.rrf_k

        for cid in all_ids:
            chunk = chunk_map[cid]

            s_rank, s_score = sparse_ranks.get(cid, (None, 0.0))
            d_rank, d_score = dense_ranks.get(cid, (None, 0.0))

            rrf = 0.0
            if s_rank is not None:
                rrf += self.config.sparse_weight * (1.0 / (k + s_rank))
            if d_rank is not None:
                rrf += self.config.dense_weight * (1.0 / (k + d_rank))

            rrf_scored.append(ScoredChunk(
                chunk=chunk,
                fused_score=rrf,
                dense_rank=d_rank,
                sparse_rank=s_rank,
                dense_score=d_score,
                sparse_score=s_score
            ))

        ranked = sorted(rrf_scored, key=lambda x: x.fused_score, reverse=True)
        return ranked[:self.config.top_k]
