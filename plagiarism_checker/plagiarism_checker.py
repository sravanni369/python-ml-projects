"""Document plagiarism checker via TF-IDF + cosine similarity.

Compares every pair of .txt files in a directory and prints a ranked list
of suspected overlaps above a threshold.

Usage:
    python plagiarism_checker.py samples/
    python plagiarism_checker.py samples/ --threshold 0.5 --top 10
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_documents(directory: Path) -> tuple[list[str], list[str]]:
    files = sorted(directory.glob("*.txt"))
    if len(files) < 2:
        raise ValueError(f"Need at least 2 .txt files in {directory}, found {len(files)}.")
    contents = [f.read_text(encoding="utf-8", errors="ignore") for f in files]
    return [f.name for f in files], contents


def similarity_matrix(documents: list[str]):
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf = vectorizer.fit_transform(documents)
    return cosine_similarity(tfidf)


def rank_pairs(names: list[str], matrix, threshold: float) -> list[tuple[str, str, float]]:
    pairs = [
        (names[i], names[j], float(matrix[i, j]))
        for i, j in combinations(range(len(names)), 2)
        if matrix[i, j] >= threshold
    ]
    return sorted(pairs, key=lambda p: p[2], reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Folder of .txt documents to compare")
    parser.add_argument("--threshold", type=float, default=0.3, help="Minimum similarity to report (0–1)")
    parser.add_argument("--top", type=int, default=20, help="Max number of pairs to show")
    args = parser.parse_args()

    names, docs = load_documents(args.directory)
    matrix = similarity_matrix(docs)
    pairs = rank_pairs(names, matrix, args.threshold)[: args.top]

    if not pairs:
        print(f"No pairs >= {args.threshold:.2f} similarity among {len(names)} documents.")
        return

    print(f"Top {len(pairs)} suspected overlaps (threshold >= {args.threshold:.2f}):\n")
    print(f"{'similarity':>10}   document A  <->  document B")
    print("-" * 72)
    for a, b, score in pairs:
        flag = "[!]" if score >= 0.8 else "   "
        print(f"{flag} {score:>7.3f}   {a}  <->  {b}")


if __name__ == "__main__":
    main()
