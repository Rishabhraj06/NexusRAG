"""Production FastAPI REST interface for NexusRAG."""

from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from nexusrag.pipeline import NexusEngine

app = FastAPI(
    title="NexusRAG API",
    description="Enterprise Multi-Agent Hybrid Retrieval-Augmented Generation Service",
    version="1.0.0"
)

# Global in-memory engine instance
engine = NexusEngine()


class IngestTextRequest(BaseModel):
    content: str = Field(..., min_length=10, description="Raw text content to chunk and index")
    source: Optional[str] = Field("api_input", description="Identifier or document name")


class IngestResponse(BaseModel):
    status: str
    chunks_created: int
    source: str


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2, description="User question or search query")


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[str]
    relevance_score: float
    is_sufficient: bool
    evaluation_reasoning: str
    model_used: str
    latency_ms: float
    retrieved_chunks: List[Dict[str, Any]]


@app.get("/health")
def healthcheck():
    return {"status": "healthy", "service": "NexusRAG"}


@app.get("/api/v1/stats")
def get_stats():
    return engine.get_stats()


@app.post("/api/v1/ingest", response_model=IngestResponse)
def ingest_text(req: IngestTextRequest):
    try:
        created = engine.ingest_text(req.content, source_name=req.source or "api_input")
        return IngestResponse(status="success", chunks_created=created, source=req.source or "api_input")
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))


@app.post("/api/v1/query", response_model=QueryResponse)
def run_query(req: QueryRequest):
    try:
        result = engine.query(req.query)
        return QueryResponse(**result)
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))
