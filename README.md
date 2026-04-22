# 🐍 Python ML Projects

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Three small, polished Python projects across classic ML, NLP, and OCR — built to demonstrate clean, production-style code on typical real-world tasks.

---

## 📁 Projects

| # | Project | Stack | What it does |
|---|---------|-------|--------------|
| 1 | [**SMS Spam Detection**](./spam_detection) | scikit-learn, NLTK, pandas | TF-IDF + Random Forest classifier (~98% accuracy on UCI SMS Spam) with a train/predict CLI |
| 2 | [**Plagiarism Checker**](./plagiarism_checker) | scikit-learn | Ranks pairs of text documents by TF-IDF cosine similarity and flags likely copies |
| 3 | [**PDF & Image Text Reader**](./pdf_image_reader) | pytesseract, PyMuPDF, pyttsx3 | OCR a PDF or image, save the text, optionally read it aloud |

Each project is self-contained with its own `README.md` and `requirements.txt`.

---

## 🚀 Quickstart

```bash
git clone https://github.com/sravanni369/python-ml-projects.git
cd python-ml-projects

# Pick a project and install its deps
cd spam_detection
pip install -r requirements.txt
python spam_detection.py --help
```

Or install everything at once from the root:

```bash
pip install -r requirements.txt
```

---

## 📜 License

MIT — see [LICENSE](./LICENSE).
