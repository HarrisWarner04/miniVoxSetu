"""
miniVoxSetu — Document Ingestion Engine (Phase 4)

Loads documents from backend/knowledge/ directory into the RAG vector store.
Replaces the need for hardcoded FAQ_DOCUMENTS.

SUPPORTED FORMATS:
  .txt  — Plain text files
  .md   — Markdown files (stripped of formatting)
  .pdf  — PDF documents (via pymupdf/fitz)

CHUNKING STRATEGY:
  Recursive character splitter that respects natural text boundaries:
  Split on: \\n\\n → \\n → ". " → " " (paragraph → sentence → word)
  chunk_size=500 chars (~125 tokens for MiniLM's 256 token window)
  chunk_overlap=50 chars (preserves context across chunk boundaries)

DEDUPLICATION:
  SHA256 hash of chunk content prevents re-indexing identical content.

USAGE:
  from ingest import DocumentIngester
  ingester = DocumentIngester()
  chunks = ingester.ingest_directory("knowledge/")
  rag_engine.initialize(external_documents=chunks)
"""

import os
import hashlib
import re
from pathlib import Path
from typing import Optional


# PDF support (optional — graceful degradation)
try:
    import fitz  # pymupdf
    PDF_AVAILABLE = True
    print("[INGEST] ✅ PDF support available (pymupdf)")
except ImportError:
    PDF_AVAILABLE = False
    print("[INGEST] ⚠️ pymupdf not installed. PDF ingestion disabled.")


# ============================================================
# TEXT CHUNKER
# ============================================================

class RecursiveCharacterSplitter:
    """
    Splits text into chunks respecting natural boundaries.

    WHY RECURSIVE:
    Instead of blindly cutting at char_limit, we try splitting at
    paragraph breaks first, then sentences, then words. This preserves
    semantic coherence within each chunk — critical for retrieval quality.

    chunk_size=500 chars is chosen because:
    - all-MiniLM-L6-v2 has 256 token limit (~4 chars/token = ~1024 chars max)
    - Smaller chunks = more precise retrieval (less noise per chunk)
    - 500 chars ≈ 2-3 sentences = a self-contained fact
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Separators ordered from most to least preferred
        self.separators = ["\n\n", "\n", ". ", " "]

    def split(self, text: str) -> list[str]:
        """Split text into overlapping chunks."""
        if len(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []

        chunks = []
        self._recursive_split(text, self.separators, chunks)
        return chunks

    def _recursive_split(self, text: str, separators: list[str], chunks: list[str]):
        """Recursively split text using the best available separator."""
        if len(text) <= self.chunk_size:
            stripped = text.strip()
            if stripped:
                chunks.append(stripped)
            return

        # Find the best separator that exists in the text
        separator = separators[0] if separators else " "
        remaining_separators = separators[1:] if len(separators) > 1 else [" "]

        # Try to split at this separator level
        parts = text.split(separator)

        current_chunk = ""
        for part in parts:
            test_chunk = current_chunk + separator + part if current_chunk else part

            if len(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                # Current chunk is full — save it
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())

                # If the individual part is too large, recurse with finer separator
                if len(part) > self.chunk_size:
                    self._recursive_split(part, remaining_separators, chunks)
                    current_chunk = ""
                else:
                    # Start new chunk with overlap from previous
                    if self.chunk_overlap > 0 and current_chunk:
                        overlap_text = current_chunk[-self.chunk_overlap:]
                        current_chunk = overlap_text + separator + part
                    else:
                        current_chunk = part

        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())


# ============================================================
# FILE PARSERS
# ============================================================

def parse_txt(file_path: str) -> str:
    """Read plain text file."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def parse_markdown(file_path: str) -> str:
    """Read markdown file and strip basic formatting."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    # Strip markdown formatting but keep content
    # Remove headers (#), bold (**), italic (*), links, images
    text = re.sub(r'#{1,6}\s*', '', text)  # Headers
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # Bold
    text = re.sub(r'\*(.+?)\*', r'\1', text)  # Italic
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # Links
    text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'\1', text)  # Images
    text = re.sub(r'`{1,3}[^`]*`{1,3}', '', text)  # Code blocks
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)  # List markers
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)  # Numbered lists
    text = re.sub(r'---+', '', text)  # Horizontal rules

    return text


def parse_pdf(file_path: str) -> str:
    """Extract text from PDF using pymupdf."""
    if not PDF_AVAILABLE:
        print(f"[INGEST] ⚠️ Cannot parse PDF (pymupdf not installed): {file_path}")
        return ""

    text_parts = []
    try:
        doc = fitz.open(file_path)
        for page in doc:
            text_parts.append(page.get_text("text"))
        doc.close()
    except Exception as e:
        print(f"[INGEST] ❌ Failed to parse PDF: {file_path}: {e}")
        return ""

    return "\n".join(text_parts)


# ============================================================
# DOCUMENT INGESTER
# ============================================================

class DocumentIngester:
    """
    Loads documents from a directory, parses them, chunks them,
    and returns clean text chunks ready for RAG embedding.

    Supports: .txt, .md, .pdf
    """

    # File extension → parser mapping
    PARSERS = {
        ".txt": parse_txt,
        ".md": parse_markdown,
        ".pdf": parse_pdf,
    }

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.splitter = RecursiveCharacterSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self._seen_hashes: set[str] = set()

    def ingest_file(self, file_path: str) -> list[dict]:
        """
        Parse a single file and return list of chunk dicts:
        [{"text": "...", "source": "filename.pdf", "chunk_index": 0}, ...]
        """
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext not in self.PARSERS:
            print(f"[INGEST] Skipping unsupported file: {path.name}")
            return []

        # Parse file to raw text
        parser = self.PARSERS[ext]
        raw_text = parser(str(path))

        if not raw_text or not raw_text.strip():
            print(f"[INGEST] Empty file: {path.name}")
            return []

        # Clean text
        raw_text = self._clean_text(raw_text)

        # Chunk the text
        chunks = self.splitter.split(raw_text)

        # Deduplicate and build metadata
        results = []
        for i, chunk_text in enumerate(chunks):
            chunk_hash = hashlib.sha256(chunk_text.encode()).hexdigest()
            if chunk_hash in self._seen_hashes:
                continue
            self._seen_hashes.add(chunk_hash)

            results.append({
                "text": chunk_text,
                "source": path.name,
                "chunk_index": i,
            })

        print(f"[INGEST] 📄 {path.name}: {len(results)} chunks")
        return results

    def ingest_directory(self, dir_path: str) -> list[str]:
        """
        Walk a directory and ingest all supported files.
        Returns flat list of chunk texts (ready for RAGEngine.initialize).
        """
        dir_path = Path(dir_path)
        if not dir_path.exists():
            print(f"[INGEST] ⚠️ Knowledge directory not found: {dir_path}")
            return []

        all_chunks = []
        supported_files = []

        # Collect supported files
        for file_path in sorted(dir_path.rglob("*")):
            if file_path.is_file() and file_path.suffix.lower() in self.PARSERS:
                # Skip README files and hidden files
                if file_path.name.lower().startswith("readme") or file_path.name.startswith("."):
                    continue
                supported_files.append(file_path)

        if not supported_files:
            print(f"[INGEST] No supported files found in {dir_path}")
            return []

        print(f"[INGEST] Found {len(supported_files)} files to ingest from {dir_path}")

        for file_path in supported_files:
            chunks = self.ingest_file(str(file_path))
            all_chunks.extend(chunks)

        # Extract just the text for RAGEngine
        texts = [chunk["text"] for chunk in all_chunks]
        print(f"[INGEST] ✅ Total: {len(texts)} unique chunks from {len(supported_files)} files")
        return texts

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalize whitespace and clean common artifacts."""
        # Collapse multiple newlines to double-newline (paragraph boundary)
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Collapse multiple spaces
        text = re.sub(r'[ \t]+', ' ', text)
        # Remove null bytes and control chars (common in PDFs)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
        return text.strip()
