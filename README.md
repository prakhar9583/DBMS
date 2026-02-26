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
