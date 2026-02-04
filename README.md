# tICA Alpha (Crypto)

This repository implements a linear time-lagged Independent Component Analysis (tICA)
alpha for cryptocurrencies using OHLCV data only, inspired by https://arxiv.org/abs/2601.11201.

## Pipeline
1. Preprocess raw long-format OHLCV parquet into wide matrices
2. Compute ATR-normalized log returns
3. Estimate rolling linear tICA weights (slowest component)
4. Apply daily smoothed weights with hysteresis signal
5. Volatility-targeted backtest with transaction costs
6. IS/OOS evaluation and parameter grid search

## How to run
1. Place `crypto_10_data.parquet` in `data/raw/`
2. Open `notebooks/run_backtest.ipynb`
3. Run cells top to bottom

Processed data is written to `data/processed/` and results to `experiments/`
(these folders are gitignored).

## Notes
- No lookahead bias (strictly rolling estimation)
- Uses only OHLCV data
- Emphasis on robustness and turnover control
