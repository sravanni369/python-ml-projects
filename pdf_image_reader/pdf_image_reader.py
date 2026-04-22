"""OCR a PDF or image and optionally read it aloud.

  Image path -> Tesseract OCR -> text
  PDF path   -> PyMuPDF rasterizes each page at 300 DPI -> Tesseract OCR -> text

Usage:
    python pdf_image_reader.py path/to/file.pdf
    python pdf_image_reader.py path/to/image.png --speak
    python pdf_image_reader.py path/to/file.pdf --out extracted.txt --speak
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import pytesseract
from PIL import Image

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}


def ocr_image(path: Path) -> str:
    with Image.open(path) as img:
        return pytesseract.image_to_string(img)


def ocr_pdf(path: Path, dpi: int = 300) -> str:
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise SystemExit("PDF support requires PyMuPDF — install with `pip install pymupdf`") from exc

    pages_text: list[str] = []
    with fitz.open(path) as doc:
        zoom = dpi / 72  # PDF default is 72 DPI
        matrix = fitz.Matrix(zoom, zoom)
        for page_num, page in enumerate(doc, start=1):
            pix = page.get_pixmap(matrix=matrix)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img).strip()
            pages_text.append(f"--- Page {page_num} ---\n{text}")
    return "\n\n".join(pages_text)


def extract_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return ocr_pdf(path)
    if suffix in IMAGE_SUFFIXES:
        return ocr_image(path)
    raise ValueError(f"Unsupported file type: {suffix}. Expected PDF or image.")


def speak(text: str) -> None:
    try:
        import pyttsx3
    except ImportError as exc:
        raise SystemExit("TTS requires pyttsx3 — install with `pip install pyttsx3`") from exc
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="PDF or image file")
    parser.add_argument("--out", type=Path, help="Optional path to save extracted text")
    parser.add_argument("--speak", action="store_true", help="Read extracted text aloud via pyttsx3")
    parser.add_argument(
        "--tesseract",
        type=str,
        default=None,
        help="Optional path to tesseract.exe (Windows users: e.g. C:\\Program Files\\Tesseract-OCR\\tesseract.exe)",
    )
    args = parser.parse_args()

    if args.tesseract:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract

    text = extract_text(args.path)
    print(text)

    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"\nSaved -> {args.out}", file=sys.stderr)

    if args.speak:
        speak(text)


if __name__ == "__main__":
    main()
