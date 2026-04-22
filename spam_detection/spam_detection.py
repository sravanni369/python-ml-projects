"""SMS spam detection using TF-IDF + Random Forest.

Trains on the UCI SMS Spam Collection and exposes:
  - `python spam_detection.py train`   -> fit model, print metrics, save artifact
  - `python spam_detection.py predict "your text here"`
"""

from __future__ import annotations

import argparse
import pickle
import re
import string
from pathlib import Path

import nltk
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ARTIFACT_PATH = Path(__file__).parent / "spam_model.pkl"
DATA_PATH = Path(__file__).parent / "data" / "sms_spam.csv"

for pkg in ("stopwords", "wordnet"):
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

STOPWORDS = set(nltk.corpus.stopwords.words("english"))
LEMMATIZER = nltk.WordNetLemmatizer()
PUNCT_TABLE = str.maketrans("", "", string.punctuation + string.digits)


def preprocess(text: str) -> str:
    text = text.lower().translate(PUNCT_TABLE)
    tokens = re.split(r"\W+", text)
    return " ".join(LEMMATIZER.lemmatize(t) for t in tokens if t and t not in STOPWORDS)


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download the UCI SMS Spam Collection "
            "(https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) "
            "and save it as data/sms_spam.csv with columns `label,text`."
        )
    df = pd.read_csv(path, encoding="latin-1")
    # Accept either the raw kaggle schema (v1,v2,...) or a cleaned one (label,text)
    if {"v1", "v2"}.issubset(df.columns):
        df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "text"})
    df = df.dropna(subset=["label", "text"]).drop_duplicates()
    df["label"] = df["label"].map({"ham": 0, "spam": 1}).astype(int)
    df["text"] = df["text"].astype(str).map(preprocess)
    return df


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
        ]
    )


def train() -> None:
    df = load_dataset(DATA_PATH)
    x_train, x_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, stratify=df["label"], random_state=42
    )
    model = build_pipeline()
    model.fit(x_train, y_train)

    preds = model.predict(x_test)
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))
    print("\nClassification report:\n", classification_report(y_test, preds, target_names=["ham", "spam"]))

    with open(ARTIFACT_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved model -> {ARTIFACT_PATH}")


def predict(text: str) -> None:
    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError("No trained model. Run `python spam_detection.py train` first.")
    with open(ARTIFACT_PATH, "rb") as f:
        model: Pipeline = pickle.load(f)
    label = model.predict([preprocess(text)])[0]
    proba = model.predict_proba([preprocess(text)])[0][label]
    print(f"{'SPAM' if label == 1 else 'HAM'} (confidence: {proba:.2%})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("train")
    p_predict = sub.add_parser("predict")
    p_predict.add_argument("text", help="SMS text to classify")
    args = parser.parse_args()

    if args.cmd == "train":
        train()
    else:
        predict(args.text)


if __name__ == "__main__":
    main()
