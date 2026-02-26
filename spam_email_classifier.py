#!/usr/bin/env python3
"""Spam Email Classifier using pure-Python Multinomial Naive Bayes.

The script expects a CSV with at least two columns:
- label: spam/ham (also supports 1/0)
- text: email body/content

It trains a bag-of-words Multinomial Naive Bayes model, evaluates on a
stratified random split, and supports classifying a single email passed with
--predict-text.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


TOKEN_RE = re.compile(r"[a-zA-Z0-9']+")


@dataclass
class EmailSample:
    label: int  # 1 = spam, 0 = ham
    text: str


def normalize_label(raw: str) -> int:
    value = raw.strip().lower()
    if value in {"1", "spam", "yes", "true"}:
        return 1
    if value in {"0", "ham", "no", "false", "legit", "not spam"}:
        return 0
    raise ValueError(f"Unsupported label value: {raw!r}")


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def load_dataset(path: str) -> List[EmailSample]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV is missing a header row.")

        fields = {name.strip().lower(): name for name in reader.fieldnames}
        label_col = fields.get("label")
        text_col = fields.get("text") or fields.get("email") or fields.get("message")
        if not label_col or not text_col:
            raise ValueError("CSV must include 'label' and 'text' (or message/email) columns.")

        samples: List[EmailSample] = []
        for row in reader:
            label_raw = (row.get(label_col) or "").strip()
            text = (row.get(text_col) or "").strip()
            if not label_raw or not text:
                continue
            samples.append(EmailSample(label=normalize_label(label_raw), text=text))

    if len(samples) < 10:
        raise ValueError("Need at least 10 usable emails for training/testing.")
    return samples


def train_test_split(samples: Sequence[EmailSample], test_ratio: float, seed: int) -> Tuple[List[EmailSample], List[EmailSample]]:
    spam = [s for s in samples if s.label == 1]
    ham = [s for s in samples if s.label == 0]

    rng = random.Random(seed)
    rng.shuffle(spam)
    rng.shuffle(ham)

    def split_class(items: List[EmailSample]) -> Tuple[List[EmailSample], List[EmailSample]]:
        cut = max(1, int(len(items) * (1 - test_ratio)))
        cut = min(cut, len(items) - 1) if len(items) > 1 else 1
        return items[:cut], items[cut:]

    spam_train, spam_test = split_class(spam)
    ham_train, ham_test = split_class(ham)

    train = spam_train + ham_train
    test = spam_test + ham_test
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


@dataclass
class NaiveBayesModel:
    log_prior_spam: float
    log_prior_ham: float
    log_prob_word_spam: Dict[str, float]
    log_prob_word_ham: Dict[str, float]
    unk_log_prob_spam: float
    unk_log_prob_ham: float


class MultinomialNaiveBayes:
    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.model: NaiveBayesModel | None = None

    def fit(self, samples: Sequence[EmailSample]) -> None:
        vocab = set()
        spam_word_counts: Dict[str, int] = {}
        ham_word_counts: Dict[str, int] = {}
        spam_docs = 0
        ham_docs = 0
        spam_total_tokens = 0
        ham_total_tokens = 0

        for sample in samples:
            tokens = tokenize(sample.text)
            if not tokens:
                continue
            if sample.label == 1:
                spam_docs += 1
                for token in tokens:
                    vocab.add(token)
                    spam_word_counts[token] = spam_word_counts.get(token, 0) + 1
                    spam_total_tokens += 1
            else:
                ham_docs += 1
                for token in tokens:
                    vocab.add(token)
                    ham_word_counts[token] = ham_word_counts.get(token, 0) + 1
                    ham_total_tokens += 1

        if spam_docs == 0 or ham_docs == 0:
            raise ValueError("Training set must include both spam and ham samples.")

        vocab_size = len(vocab)
        spam_denom = spam_total_tokens + self.alpha * vocab_size
        ham_denom = ham_total_tokens + self.alpha * vocab_size

        log_prob_word_spam = {
            w: math.log((spam_word_counts.get(w, 0) + self.alpha) / spam_denom) for w in vocab
        }
        log_prob_word_ham = {
            w: math.log((ham_word_counts.get(w, 0) + self.alpha) / ham_denom) for w in vocab
        }

        total_docs = spam_docs + ham_docs
        self.model = NaiveBayesModel(
            log_prior_spam=math.log(spam_docs / total_docs),
            log_prior_ham=math.log(ham_docs / total_docs),
            log_prob_word_spam=log_prob_word_spam,
            log_prob_word_ham=log_prob_word_ham,
            unk_log_prob_spam=math.log(self.alpha / spam_denom),
            unk_log_prob_ham=math.log(self.alpha / ham_denom),
        )

    def predict_one(self, text: str) -> int:
        if self.model is None:
            raise ValueError("Model is not trained.")
        tokens = tokenize(text)

        spam_score = self.model.log_prior_spam
        ham_score = self.model.log_prior_ham
        for token in tokens:
            spam_score += self.model.log_prob_word_spam.get(token, self.model.unk_log_prob_spam)
            ham_score += self.model.log_prob_word_ham.get(token, self.model.unk_log_prob_ham)
        return 1 if spam_score > ham_score else 0

    def predict(self, texts: Sequence[str]) -> List[int]:
        return [self.predict_one(t) for t in texts]


def evaluate(y_true: Sequence[int], y_pred: Sequence[int]) -> Tuple[float, float, float, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    accuracy = (tp + tn) / max(1, len(y_true))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)
    return accuracy, precision, recall, f1


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a spam email classifier (Multinomial Naive Bayes).")
    parser.add_argument("--csv", required=True, help="Path to CSV containing label and text columns.")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Test split ratio (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the split")
    parser.add_argument(
        "--predict-text",
        type=str,
        default="",
        help="Optional: classify a single email text after training.",
    )
    args = parser.parse_args()

    if not (0.05 <= args.test_ratio <= 0.5):
        raise ValueError("--test-ratio should be between 0.05 and 0.5")

    samples = load_dataset(args.csv)
    train, test = train_test_split(samples, args.test_ratio, args.seed)

    model = MultinomialNaiveBayes(alpha=1.0)
    model.fit(train)

    y_true = [s.label for s in test]
    y_pred = model.predict([s.text for s in test])
    accuracy, precision, recall, f1 = evaluate(y_true, y_pred)

    print("Model: Multinomial Naive Bayes")
    print(f"Train samples: {len(train)} | Test samples: {len(test)}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    if args.predict_text.strip():
        pred = model.predict_one(args.predict_text)
        label = "spam" if pred == 1 else "ham"
        print(f"Prediction for input text: {label}")


if __name__ == "__main__":
    main()
