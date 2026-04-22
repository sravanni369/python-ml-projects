# 🔊 PDF & Image Text Reader That Speaks

OCR a PDF or image file, print the extracted text, and optionally read it aloud.

## Pipeline

- **Image** → `pytesseract.image_to_string`
- **PDF** → `PyMuPDF` renders each page at 300 DPI → `pytesseract` OCR → per-page text
- **TTS** (optional) → `pyttsx3` offline text-to-speech

## Prerequisites

1. **Tesseract OCR** — install the binary for your OS:
   - Windows: <https://github.com/UB-Mannheim/tesseract/wiki>
   - macOS: `brew install tesseract`
   - Linux: `sudo apt install tesseract-ocr`

   On Windows, either add the install dir to `PATH` or pass `--tesseract "C:\Program Files\Tesseract-OCR\tesseract.exe"`.

2. **Python deps**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

```bash
# Image file
python pdf_image_reader.py samples/sample_image.png

# PDF — all pages, stamp with page headers
python pdf_image_reader.py samples/sample_document.pdf

# Save extracted text to a file
python pdf_image_reader.py samples/sample_document.pdf --out extracted.txt

# Read aloud after extracting
python pdf_image_reader.py samples/sample_image.png --speak

# Windows: point at a non-PATH Tesseract install
python pdf_image_reader.py samples/x.pdf --tesseract "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

## Why PyMuPDF instead of Wand/ImageMagick?

The original recipe uses `wand` (ImageMagick). On Windows that's a multi-step native install with Ghostscript. `PyMuPDF` is a single `pip install pymupdf` — zero native deps, and its rasterizer is fast.
