#!/usr/bin/env python3
"""Stock Price Predictor using pure-Python linear regression.

This script trains a next-day closing-price predictor from historical CSV data.
Expected CSV columns: Date, Close (case-insensitive supported).
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple


@dataclass
class Dataset:
    dates: List[str]
    closes: List[float]


def load_dataset(path: str) -> Dataset:
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV file is missing a header row.")

        normalized = {name.strip().lower(): name for name in reader.fieldnames}
        date_col = normalized.get("date")
        close_col = normalized.get("close")
        if not date_col or not close_col:
            raise ValueError("CSV must contain 'Date' and 'Close' columns.")

        dates: List[str] = []
        closes: List[float] = []
        for row in reader:
            date = row.get(date_col, "").strip()
            close_raw = row.get(close_col, "").strip()
            if not date or not close_raw:
                continue
            try:
                close = float(close_raw)
            except ValueError:
                continue
            dates.append(date)
            closes.append(close)

    if len(closes) < 10:
        raise ValueError("Not enough usable rows. Provide at least 10 close-price rows.")
    return Dataset(dates=dates, closes=closes)


def build_supervised(closes: Sequence[float], lag: int) -> Tuple[List[List[float]], List[float]]:
    if lag < 1:
        raise ValueError("Lag must be >= 1")
    if len(closes) <= lag:
        raise ValueError(f"Need more rows than lag ({lag}).")

    x: List[List[float]] = []
    y: List[float] = []
    for i in range(lag, len(closes)):
        features = [1.0] + [closes[i - j] for j in range(1, lag + 1)]
        x.append(features)
        y.append(closes[i])
    return x, y


def transpose(matrix: Sequence[Sequence[float]]) -> List[List[float]]:
    return [list(col) for col in zip(*matrix)]


def matmul(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> List[List[float]]:
    b_t = transpose(b)
    return [[sum(x * y for x, y in zip(row, col)) for col in b_t] for row in a]


def matvec(a: Sequence[Sequence[float]], v: Sequence[float]) -> List[float]:
    return [sum(x * y for x, y in zip(row, v)) for row in a]


def solve_linear_system(a: List[List[float]], b: List[float]) -> List[float]:
    n = len(a)
    aug = [row[:] + [rhs] for row, rhs in zip(a, b)]

    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            raise ValueError("Singular matrix encountered while training model.")
        aug[col], aug[pivot] = aug[pivot], aug[col]

        pivot_val = aug[col][col]
        aug[col] = [v / pivot_val for v in aug[col]]

        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            aug[r] = [rv - factor * cv for rv, cv in zip(aug[r], aug[col])]

    return [aug[i][-1] for i in range(n)]


def fit_ridge_regression(x: List[List[float]], y: List[float], ridge: float = 1e-6) -> List[float]:
    x_t = transpose(x)
    xtx = matmul(x_t, x)
    xty = matvec(x_t, y)

    for i in range(len(xtx)):
        xtx[i][i] += ridge

    return solve_linear_system(xtx, xty)


def predict_row(weights: Sequence[float], row: Sequence[float]) -> float:
    return sum(w * x for w, x in zip(weights, row))


def evaluate(y_true: Sequence[float], y_pred: Sequence[float]) -> Tuple[float, float, float]:
    errors = [t - p for t, p in zip(y_true, y_pred)]
    mae = sum(abs(e) for e in errors) / len(errors)
    rmse = math.sqrt(sum(e * e for e in errors) / len(errors))

    mean_y = sum(y_true) / len(y_true)
    ss_tot = sum((y - mean_y) ** 2 for y in y_true)
    ss_res = sum((t - p) ** 2 for t, p in zip(y_true, y_pred))
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    return mae, rmse, r2


def train_test_split(
    x: List[List[float]], y: List[float], test_ratio: float
) -> Tuple[List[List[float]], List[float], List[List[float]], List[float]]:
    split = max(1, int(len(x) * (1 - test_ratio)))
    split = min(split, len(x) - 1)
    return x[:split], y[:split], x[split:], y[split:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict stock close prices using linear regression.")
    parser.add_argument("--csv", required=True, help="Path to CSV with Date and Close columns.")
    parser.add_argument("--lag", type=int, default=5, help="Number of previous days used as features.")
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Fraction of data for testing (chronological split).",
    )
    args = parser.parse_args()

    dataset = load_dataset(args.csv)
    x, y = build_supervised(dataset.closes, args.lag)
    x_train, y_train, x_test, y_test = train_test_split(x, y, args.test_ratio)

    weights = fit_ridge_regression(x_train, y_train)
    preds = [predict_row(weights, row) for row in x_test]
    mae, rmse, r2 = evaluate(y_test, preds)

    recent_window = dataset.closes[-args.lag :]
    next_day_row = [1.0] + list(reversed(recent_window))
    next_day_pred = predict_row(weights, next_day_row)

    print("Model: Linear Regression (normal equation + tiny ridge)")
    print(f"Train samples: {len(x_train)} | Test samples: {len(x_test)}")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")
    print(f"Predicted next close price: {next_day_pred:.4f}")


if __name__ == "__main__":
    main()
