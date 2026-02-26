# DBMS

## Stock Price Predictor

This repository now includes `stock_price_predictor.py`, a pure-Python stock price predictor that trains a **linear regression** model on historical close prices.

### What it does
- Loads historical data from a CSV file (`Date`, `Close` columns).
- Builds lag features (for example, previous 5 closing prices).
- Trains linear regression with a tiny ridge penalty for numerical stability.
- Evaluates on a chronological test split with:
  - MAE
  - RMSE
  - R²
- Predicts the next day's closing price.

### Run
```bash
python stock_price_predictor.py --csv your_stock_data.csv --lag 5 --test-ratio 0.2
```

### CSV format
```csv
Date,Close
2024-01-01,187.15
2024-01-02,189.03
...
```

You can change `--lag` to control how many prior days are used as model features.

---

## Spam Email Classifier

This repository also includes `spam_email_classifier.py`, a pure-Python spam filter based on **Multinomial Naive Bayes**.

### What it does
- Loads labeled emails from a CSV with `label` and `text` columns.
- Tokenizes email text into a bag-of-words representation.
- Trains a Multinomial Naive Bayes classifier with Laplace smoothing.
- Evaluates on a stratified train/test split using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
- Optionally predicts whether a custom email text is spam or ham.

### Run
```bash
python spam_email_classifier.py --csv emails.csv --test-ratio 0.2 --seed 42
```

### Classify one email after training
```bash
python spam_email_classifier.py --csv emails.csv --predict-text "Congratulations! You won a free gift card."
```

### CSV format
```csv
label,text
ham,"Hi team, attached is the meeting agenda for tomorrow."
spam,"Winner! Click now to claim your prize"
...
```

Supported label values include `spam`/`ham` and `1`/`0`.
