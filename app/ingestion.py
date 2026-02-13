from io import BytesIO
from pathlib import Path

from PIL import Image
from pypdf import PdfReader
import pytesseract


def extract_text(filename: str, content: bytes) -> tuple[str, str]:
    suffix = Path(filename).suffix.lower()

    if suffix in {".txt", ".md", ".csv", ".log"}:
        return content.decode("utf-8", errors="ignore"), "text"

    if suffix == ".pdf":
        pdf = PdfReader(BytesIO(content))
        pages = [page.extract_text() or "" for page in pdf.pages]
        return "\n".join(pages), "pdf"

    if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}:
        image = Image.open(BytesIO(content))
        text = pytesseract.image_to_string(image)
        return text, "image"

    raise ValueError(f"Unsupported file type: {suffix}")


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> list[str]:
    if not text.strip():
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks
