"""Document parsing, metadata tracking, and semantic text chunking."""

import re
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from nexusrag.config import ChunkingConfig


@dataclass
class DocumentChunk:
    chunk_id: str
    doc_id: str
    source: str
    content: str
    chunk_index: int
    char_length: int
    metadata: Dict[str, Any]


class DocumentParser:
    """Extracts raw text and metadata from files (PDF, Markdown, TXT, Code)."""

    @staticmethod
    def parse_file(file_path: Path) -> str:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {path}")

        ext = path.suffix.lower()

        if ext == ".pdf":
            return DocumentParser._parse_pdf(path)
        else:
            # Default text read for .md, .txt, .json, .py, etc.
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()

    @staticmethod
    def _parse_pdf(path: Path) -> str:
        try:
            import pypdf
            reader = pypdf.PdfReader(str(path))
            pages = []
            for idx, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append(f"[Page {idx + 1}]\n{text}")
            return "\n\n".join(pages)
        except Exception as err:
            raise RuntimeError(f"Failed to extract text from PDF: {err}")


class SemanticChunker:
    """Splits raw document text into overlapping semantic passages."""

    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()

    def chunk_text(self, text: str, source: str, doc_id: Optional[str] = None) -> List[DocumentChunk]:
        clean_text = self._sanitize(text)
        if not clean_text:
            return []

        doc_hash = doc_id or hashlib.sha256(source.encode()).hexdigest()[:12]
        paragraphs = clean_text.split("\n\n")

        raw_chunks = []
        current_chunk = []
        current_len = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_len = len(para)
            if current_len + para_len > self.config.chunk_size and current_chunk:
                combined = " ".join(current_chunk)
                if len(combined) >= self.config.min_chunk_len:
                    raw_chunks.append(combined)

                # Keep overlap from the end of current chunk
                overlap_text = combined[-self.config.chunk_overlap:] if self.config.chunk_overlap > 0 else ""
                current_chunk = [overlap_text, para] if overlap_text else [para]
                current_len = len(overlap_text) + para_len
            else:
                current_chunk.append(para)
                current_len += para_len

        if current_chunk:
            combined = " ".join(current_chunk).strip()
            if len(combined) >= self.config.min_chunk_len:
                raw_chunks.append(combined)

        # Build structured DocumentChunk objects
        chunks: List[DocumentChunk] = []
        for idx, content in enumerate(raw_chunks):
            cid = f"{doc_hash}_c{idx}"
            chunks.append(DocumentChunk(
                chunk_id=cid,
                doc_id=doc_hash,
                source=source,
                content=content,
                chunk_index=idx,
                char_length=len(content),
                metadata={"source_file": Path(source).name}
            ))

        return chunks

    @staticmethod
    def _sanitize(text: str) -> str:
        # Normalize whitespace and excessive line breaks
        text = re.sub(r"\r\n", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
