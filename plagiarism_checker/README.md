# 📝 Plagiarism Checker

Ranks pairs of text documents by similarity using TF-IDF vectors and cosine distance.

## How it works

1. Load every `.txt` file in the supplied directory.
2. Vectorize with TF-IDF (unigrams + bigrams, English stopwords removed).
3. Compute the full pairwise cosine-similarity matrix.
4. Report the top-N pairs above a similarity threshold.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python plagiarism_checker.py samples/
python plagiarism_checker.py samples/ --threshold 0.5 --top 10
```

## Sample output

```
Top 3 suspected overlaps (threshold >= 0.30):

similarity   document A  <->  document B
------------------------------------------------------------------------
[!]   0.912   essay_alice.txt  <->  essay_alice_copy.txt
      0.584   essay_alice.txt  <->  essay_bob.txt
      0.402   essay_bob.txt    <->  essay_carol.txt
```

Pairs above **0.80** are flagged with `[!]` as likely direct copies.

## Notes

- Cosine similarity on TF-IDF catches near-duplicate prose and heavy paraphrasing of the same vocabulary, but will miss deep paraphrasing that swaps all the content words. For that, move to sentence-transformer embeddings.
- Short documents (< ~50 words) produce noisy scores — prefer longer inputs.
