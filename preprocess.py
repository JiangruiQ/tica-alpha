"""
preprocess.py

Helpers to convert a raw long-format parquet (crypto_10_data.parquet)
into the canonical wide parquets used by the pipeline:
 - close_wide.parquet, high_wide.parquet, low_wide.parquet, open_wide.parquet, volume_wide.parquet
 - returns_normalized_wide.parquet (ATR-normalized log returns)
 - asset_diagnostics.parquet

Usage (notebook):
>>> from src.preprocess import prepare_from_parquet
>>> prepare_from_parquet("/path/to/crypto_10_data.parquet", out_dir="data/processed")
"""

from typing import Optional
import os
import pandas as pd
import numpy as np


def compute_log_returns(close: pd.DataFrame) -> pd.DataFrame:
    return np.log(close).diff().iloc[1:]


def compute_true_range(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.DataFrame(np.maximum(np.maximum(tr1.values, tr2.values), tr3.values),
                      index=tr1.index, columns=tr1.columns)
    return tr


def compute_atr(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    tr = compute_true_range(high, low, close)
    atr = tr.rolling(window=period, min_periods=1).mean()
    return atr


def prepare_from_parquet(
    raw_parquet_path: str,
    out_dir: str = "data/processed",
    time_col: str = "time",
    ticker_col: str = "ticker",
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: str = "volume",
    atr_period: int = 14,
    min_avail: float = 0.90,
    ffill_limit: int = 3,
    verbose: bool = True,
):
    """
    Read raw parquet (long), pivot to wide OHLCV and compute ATR-normalized returns.

    Parameters
    ----------
    raw_parquet_path : path to the user's uploaded parquet (long-format)
    out_dir : directory to write processed wide parquets
    atr_period : ATR window in days
    min_avail : minimum fraction of available close values to keep an asset
    ffill_limit : forward-fill limit (int days) to impute small gaps
    """

    os.makedirs(out_dir, exist_ok=True)

    # load
    if verbose:
        print("Loading raw parquet:", raw_parquet_path)
    df = pd.read_parquet(raw_parquet_path)

    # Ensure expected columns exist
    expected = {time_col, ticker_col, open_col, high_col, low_col, close_col, volume_col}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns in raw parquet: {missing}")

    # Parse time
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values([time_col, ticker_col])

    # Pivot to wide tables
    if verbose: print("Pivoting to wide tables...")
    close_wide = df.pivot(index=time_col, columns=ticker_col, values=close_col).sort_index()
    open_wide = df.pivot(index=time_col, columns=ticker_col, values=open_col).sort_index()
    high_wide  = df.pivot(index=time_col, columns=ticker_col, values=high_col).sort_index()
    low_wide   = df.pivot(index=time_col, columns=ticker_col, values=low_col).sort_index()
    vol_wide   = df.pivot(index=time_col, columns=ticker_col, values=volume_col).sort_index()

    # Basic diagnostics: availability, median volume
    avail = close_wide.notnull().mean()
    med_vol = vol_wide.median()
    diag = pd.DataFrame({"availability": avail, "median_volume": med_vol}).sort_values("availability", ascending=False)

    # Filter assets by availability
    keep = avail[avail >= min_avail].index.tolist()
    if verbose:
        print(f"Keeping {len(keep)}/{len(avail)} assets with availability >= {min_avail}")

    # small-gap imputation (limit forward/back fill to ffill_limit)
    if verbose: print("Imputing small gaps (ffill/bfill) ...")
    close_ff = close_wide.fillna(method="ffill", limit=ffill_limit).fillna(method="bfill", limit=ffill_limit)
    high_ff  = high_wide.fillna(method="ffill", limit=ffill_limit).fillna(method="bfill", limit=ffill_limit)
    low_ff   = low_wide.fillna(method="ffill", limit=ffill_limit).fillna(method="bfill", limit=ffill_limit)
    open_ff  = open_wide.fillna(method="ffill", limit=ffill_limit).fillna(method="bfill", limit=ffill_limit)
    vol_ff   = vol_wide.fillna(0.0)

    close_proc = close_ff[keep].copy()
    high_proc = high_ff[keep].copy()
    low_proc = low_ff[keep].copy()
    open_proc = open_ff[keep].copy()
    vol_proc = vol_ff[keep].copy()
    diag_proc = diag.loc[keep]

    # Compute returns (log returns)
    if verbose: print("Computing log returns...")
    returns = compute_log_returns(close_proc)

    # Compute ATR, convert to return units, normalize returns
    if verbose: print(f"Computing ATR (period={atr_period}) and normalizing returns...")
    atr = compute_atr(high_proc, low_proc, close_proc, period=atr_period)
    # align atr to returns index: atr indexed by same times as tr (has first row where prev exists)
    atr = atr.reindex(returns.index).ffill().bfill()
    atr_ret = atr.divide(close_proc.reindex(atr.index))
    returns_norm = returns.divide(atr_ret).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Save outputs
    if verbose: print("Saving processed files to:", out_dir)
    close_proc.to_parquet(os.path.join(out_dir, "close_wide.parquet"))
    high_proc.to_parquet(os.path.join(out_dir, "high_wide.parquet"))
    low_proc.to_parquet(os.path.join(out_dir, "low_wide.parquet"))
    open_proc.to_parquet(os.path.join(out_dir, "open_wide.parquet"))
    vol_proc.to_parquet(os.path.join(out_dir, "volume_wide.parquet"))
    returns_norm.to_parquet(os.path.join(out_dir, "returns_normalized_wide.parquet"))
    diag_proc.to_parquet(os.path.join(out_dir, "asset_diagnostics.parquet"))

    if verbose:
        print("Saved:")
        print(" - close_wide.parquet etc.")
        print(" - returns_normalized_wide.parquet")
        print(" - asset_diagnostics.parquet")
        print("Processed tickers:", list(keep))

    return {
        "close": close_proc,
        "high": high_proc,
        "low": low_proc,
        "open": open_proc,
        "volume": vol_proc,
        "returns_normalized": returns_norm,
        "diagnostics": diag_proc,
    }


# small wrapper for notebook convenience
if __name__ == "__main__":
    # quick CLI-like convenience if run as script
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="Raw parquet file (long-format)")
    parser.add_argument("--out", "-o", default="data/processed", help="Processed output dir")
    parser.add_argument("--atr", type=int, default=14, help="ATR period")
    parser.add_argument("--min_avail", type=float, default=0.9, help="Min availability fraction")
    args = parser.parse_args()
    prepare_from_parquet(args.input, out_dir=args.out, atr_period=args.atr, min_avail=args.min_avail)
