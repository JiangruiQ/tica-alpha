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



## Flow chart
                       ┌────────────────────────┐
                       │ 1) Raw OHLCV (wide)    │
                       └────────────┬───────────┘
                                    │
                                    ▼
                       ┌────────────────────────┐
                       │ 2) Preprocess & Normalize │
                       │    (ATR-normalize returns)│
                       └────────────┬───────────┘
                                    │
                                    ▼
                       ┌────────────────────────┐
                       │ 3) Rolling tICA weights │
                       │    compute_rolling_tica_weights()
                       └────────────┬───────────┘
                                    │
                                    ▼
                       ┌────────────────────────┐
                       │ 4) Compute factor (t)  │
                       │ factor_t = w_tᵀ(x_t - μ_window)
                       │ compute_factor_series_from_weights()
                       └────────────┬───────────┘
                                    │
                                    ▼
                       ┌────────────────────────┐
                       │ 5) Smooth weights (daily) │
                       │ weights_sm = smooth_weights(weights_raw)
                       └────────────┬───────────┘
                                    │
                                    ▼
                       ┌────────────────────────┐
                       │ 6) Hysteresis signal   │
                       │ direction ∈ {-1,0,+1}  │
                       │ hysteresis_signal(factor)
                       └────────────┬───────────┘
                                    │
                                    ▼
                       ┌────────────────────────┐
                       │ 7) Build positions     │
                       │ pos_target_t = direction_t * weights_sm_t
                       │ build_positions()
                       └────────────┬───────────┘
                                    │
                                    ▼
                       ┌────────────────────────┐
                       │ 8) Rebalance & PnL     │
                       │ pos_prev = pos_target_{t-1}               <---|
                       │ turnover_t = ∑ |pos_target_t - pos_prev|  |  |
                       │ costs = tc * turnover                      |  |
                       │ daily_raw = pos_prev ⋅ returns_t           |  |
                       │ daily_net = daily_raw - costs              |  |
                       │ daily_vt = daily_net * scaling (vol target)|  |
                       └─────────────────────────────────────────────┘

- `w_t` = tICA slowest eigenvector (L1-normalized)
- `x_t` = today's normalized returns vector
- `μ_window` = mean of returns over tICA window
- `direction_t` = -1 / 0 / +1 from hysteresis
- `weights_sm_t` = smoothed L1-normalized weight vector
- `pos_target_t` = target allocation (signed, sum|·| = 1 when direction ±1)
