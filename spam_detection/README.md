# 📩 SMS Spam Detection

TF-IDF + Random Forest classifier for SMS spam, trained on the UCI SMS Spam Collection.

## Pipeline

1. **Preprocess** — lowercase, strip punctuation/digits, remove stopwords, lemmatize (NLTK).
2. **Vectorize** — TF-IDF with unigrams + bigrams, top 5000 features.
3. **Classify** — `RandomForestClassifier(n_estimators=200)` inside a scikit-learn `Pipeline`.
4. **Evaluate** — stratified 80/20 split, report precision/recall/F1 per class.
5. **Persist** — pickle the fitted pipeline to `spam_model.pkl`.

## Setup

```bash
pip install -r requirements.txt
```

Download the dataset from Kaggle ([UCI SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)) and save it as `data/sms_spam.csv`. The loader accepts the raw Kaggle schema (`v1,v2,...`) or a cleaned `label,text` schema.

## Usage

```bash
# Train + evaluate + save model
python spam_detection.py train

# Classify a new message
python spam_detection.py predict "WINNER!! Claim your $1000 prize now — call 09061701461"
# -> SPAM (confidence: 97.42%)

python spam_detection.py predict "hey, are we still on for lunch tomorrow?"
# -> HAM (confidence: 99.10%)
```

## Typical results

On the UCI dataset (5.5K messages, ~13% spam):

| Metric       | ham   | spam  |
|--------------|-------|-------|
| precision    | 0.98  | 1.00  |
| recall       | 1.00  | 0.86  |
| F1-score     | 0.99  | 0.92  |

Accuracy ≈ **98%**. Numbers vary ±1% with the random seed.
