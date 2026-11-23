"""Load curriculum documents (.txt, .md, .pdf) from a folder and chunk them."""
from pathlib import Path

from gradient.rag import Chunk, chunk_text

SUPPORTED = {".txt", ".md", ".pdf"}


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:  # pragma: no cover
        raise ImportError("Reading PDFs needs pypdf: pip install pypdf") from e
    return "\n".join((page.extract_text() or "") for page in PdfReader(str(path)).pages)


def read_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return _read_pdf(path)
    return path.read_text(encoding="utf-8")


def load_documents(directory: str | Path, size: int = 80, overlap: int = 20) -> list[Chunk]:
    """Chunk every supported file in `directory`. README files are skipped (they are docs, not curriculum)."""
    chunks: list[Chunk] = []
    for path in sorted(Path(directory).iterdir()):
        if path.suffix.lower() not in SUPPORTED or path.stem.lower().startswith("readme"):
            continue
        text = read_document(path).strip()
        if text:
            chunks.extend(chunk_text(text, source=path.name, size=size, overlap=overlap))
    return chunks
