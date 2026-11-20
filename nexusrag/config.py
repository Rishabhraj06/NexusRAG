"""Configuration and system parameters for NexusRAG."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ChunkingConfig:
    chunk_size: int = 500
    chunk_overlap: int = 80
    min_chunk_len: int = 40


@dataclass
class RetrieverConfig:
    top_k: int = 4
    rrf_k: int = 60
    dense_weight: float = 0.5
    sparse_weight: float = 0.5


@dataclass
class AgentConfig:
    relevance_threshold: float = 0.65
    max_expansions: int = 2
    model_name: str = "gemini-2.5-flash"


@dataclass
class NexusSettings:
    storage_dir: Path = field(default_factory=lambda: Path(".nexus_storage"))
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
