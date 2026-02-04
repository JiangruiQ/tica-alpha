"""
tica_alpha.py
A compact module that provides reusable functions for:
- loading wide OHLCV parquet / csv
- computing returns & ATR-normalization
- rolling linear tICA weight estimation
- factor computation, daily smoothing of weights
- hysteresis signal, position construction, turnover & costs
- vol-target scaling and performance metrics

Author: Jiangrui
"""

from typing import Tuple, Optional
import os
import math
import numpy as np
import pandas as pd
from scipy import linalg
import matplotlib.pyplot as plt
import datetime as dt

# ---------------------------
# I/O & basic helpers
# ---------------------------

def load_wide_parquet(processed_dir: str = "data/processed") -> pd.DataFrame:
    """
    Load normalized returns wide parquet if present, otherwise attempt to load close/high/low and compute.
    Expects returns_normalized_wide.parquet or close_wide.parquet/high_wide.parquet/low_wide.parquet.
    Returns:
        returns_norm: DataFrame indexed by datetime, columns=tickers
    """
    rn_path = os.path.join(processed_dir, "returns_normalized_wide.parquet")
    if os.path.exists(rn_path):
        return pd.read_parquet(rn_path)
    # Fallback: try to build returns_norm from close/high/low
    close_p = os.path.join(processed_dir, "close_wide.parquet")
    high_p = os.path.join(processed_dir, "high_wide.parquet")
    low_p = os.path.join(processed_dir, "low_wide.parquet")
    if all(os.path.exists(p) for p in (close_p, high_p, low_p)):
        close = pd.read_parquet(close_p); high = pd.read_parquet(high_p); low = pd.read_parquet(low_p)
        # ensure datetimes
        for df in (close, high, low):
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
        # compute returns and ATR->normalize (14-day default)
        returns = compute_log_returns(close)
        atr = compute_atr(high, low, close, period=14).reindex(returns.index).ffill().bfill()
        atr_ret = atr.divide(close.reindex(atr.index))
        returns_norm = returns.divide(atr_ret).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return returns_norm
    raise FileNotFoundError("Could not find returns_normalized_wide.parquet or close/high/low in processed dir.")

def save_series_parquet(series: pd.Series, path: str):
    series.to_parquet(path)

def save_df_parquet(df: pd.DataFrame, path: str):
    df.to_parquet(path)

# ---------------------------
# Feature engineering
# ---------------------------

def compute_log_returns(close: pd.DataFrame) -> pd.DataFrame:
    """Close-to-close log returns. Drops the first NaN row."""
    r = np.log(close).diff().iloc[1:]
    r.index = pd.to_datetime(r.index)
    return r

def compute_true_range(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.DataFrame(np.maximum(np.maximum(tr1.values, tr2.values), tr3.values),
                      index=tr1.index, columns=tr1.columns)
    return tr

def compute_atr(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Simple moving average ATR (period days). Returns same index as true range (first row includes t where prev exists)."""
    tr = compute_true_range(high, low, close)
    atr = tr.rolling(window=period, min_periods=1).mean()
    return atr

def normalize_returns_by_atr(returns: pd.DataFrame, atr: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    """
    Convert ATR to return units (ATR/close) then divide returns by atr_return.
    Returns dimensionless normalized returns.
    """
    atr_ret = atr.divide(close.reindex(atr.index))
    returns = returns.reindex(atr_ret.index).fillna(0.0)
    rn = returns.divide(atr_ret).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return rn

# ---------------------------
# tICA solver & rolling estimator
# ---------------------------

def solve_generalized_eigen(CT: np.ndarray, C0: np.ndarray, reg_eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve CT v = lambda C0 v. Return eigenvalues (desc) and eigenvectors (columns).
    Adds fallback whitening if direct solver fails.
    """
    try:
        eigvals, eigvecs = linalg.eigh(CT, C0)
        order = np.argsort(eigvals)[::-1]
        return eigvals[order], eigvecs[:, order]
    except Exception:
        sqrtC0 = linalg.sqrtm(C0)
        inv_sqrt = linalg.inv(sqrtC0)
        M = inv_sqrt @ CT @ inv_sqrt
        eigvals_std, eigvecs_std = linalg.eigh(M)
        order = np.argsort(eigvals_std)[::-1]
        eigvecs = inv_sqrt @ eigvecs_std[:, order]
        return eigvals_std[order], eigvecs

def compute_rolling_tica_weights(returns: pd.DataFrame, window: int = 500, lag_T: int = 7,
                                 reg_eps: float = 1e-8, verbose: bool = False) -> pd.DataFrame:
    """
    Compute daily rolling tICA slowest eigenvector (L1-normalized) for each date where estimation is possible.
    returns: T x N DataFrame
    """
    dates = returns.index
    vals = returns.values  # shape (T, N)
    T, N = vals.shape
    weights = pd.DataFrame(index=dates, columns=returns.columns, dtype=float)

    for i in range(window, T):
        Xw = vals[i - window:i, :]
        Xc = Xw - Xw.mean(axis=0)
        C0 = (Xc.T @ Xc) / Xc.shape[0]
        C0 = C0 + reg_eps * np.eye(N)
        if lag_T >= Xc.shape[0]:
            continue
        X1 = Xc[:-lag_T, :]
        X2 = Xc[lag_T:, :]
        CT = 0.5 * (X2.T @ X1) / (Xc.shape[0] - lag_T)
        CT = 0.5 * (CT + CT.T)
        eigvals, eigvecs = solve_generalized_eigen(CT, C0, reg_eps=reg_eps)
        w = np.real(eigvecs[:, 0])
        w = w / (np.sum(np.abs(w)) + 1e-12)
        weights.iloc[i] = w
        if verbose and (i % 250 == 0):
            print(f"tICA progress: {i}/{T}")
    return weights

def compute_factor_series_from_weights(returns: pd.DataFrame, weights: pd.DataFrame, window: int) -> pd.Series:
    """
    For each day where weights exists, compute y_t = w_t^T (x_t - mean_in_window)
    returns and weights aligned by index.
    """
    dates = returns.index
    vals = returns.values
    T, N = vals.shape
    f = np.full(T, np.nan)
    for i in range(window, T):
        wvec = weights.iloc[i]
        if wvec.isna().all():
            continue
        Xw = vals[i - window:i, :]
        mean_col = Xw.mean(axis=0)
        x_today = vals[i, :] - mean_col
        f[i] = float(np.dot(wvec.values, x_today))
    return pd.Series(f, index=dates)

# ---------------------------
# Weight smoothing, signal, positions
# ---------------------------

def smooth_weights(weights_raw: pd.DataFrame, alpha: float = 0.2) -> pd.DataFrame:
    """
    Exponential smoothing across days (alpha in (0,1]); smaller alpha -> slower adaptation.
    We re-normalize so L1 sum(abs)=1 every day (if not all zeros).
    """
    dates = weights_raw.index
    sm = pd.DataFrame(index=dates, columns=weights_raw.columns, dtype=float)
    w_prev = None
    for dt in dates:
        w_new = weights_raw.loc[dt]
        if w_new.isna().all():
            sm.loc[dt] = w_prev if w_prev is not None else 0.0
            continue
        w_new = w_new.astype(float).values
        denom = np.sum(np.abs(w_new)) + 1e-12
        w_new = w_new / denom
        if w_prev is None:
            w_sm = w_new
        else:
            w_sm = alpha * w_new + (1.0 - alpha) * w_prev
            w_sm = w_sm / (np.sum(np.abs(w_sm)) + 1e-12)
        sm.loc[dt] = w_sm
        w_prev = w_sm
    return sm

def hysteresis_signal(factor: pd.Series, ema_span: int = 20, hysteresis_k: float = 0.06, vol_window: int = 60) -> pd.Series:
    """
    Compute direction series with hysteresis:
      - enter long when EMA(f) > +k * rolling_std
      - enter short when EMA(f) < -k * rolling_std
      - exit long when EMA(f) < 0
      - exit short when EMA(f) > 0
    Returns -1, 0, +1 series aligned to factor index.
    """
    f_ema = factor.ewm(span=ema_span, adjust=False).mean()
    f_std = factor.rolling(vol_window).std()
    h = hysteresis_k * f_std
    dates = factor.index
    direction = pd.Series(0, index=dates)
    state = 0
    for i, dt in enumerate(dates):
        f = f_ema.iloc[i]
        th = h.iloc[i]
        if np.isnan(f) or np.isnan(th):
            direction.iloc[i] = state
            continue
        if state == 0:
            if f > +th:
                state = +1
            elif f < -th:
                state = -1
        elif state == +1:
            if f < 0:
                state = 0
        elif state == -1:
            if f > 0:
                state = 0
        direction.iloc[i] = state
    return direction

def build_positions(direction: pd.Series, weights_smoothed: pd.DataFrame) -> pd.DataFrame:
    """
    Construct daily target positions = direction_t * weights_smoothed_t
    """
    dates = direction.index
    returns_idx = weights_smoothed.index
    # align indices (weights already aligned)
    pos = pd.DataFrame(0.0, index=weights_smoothed.index, columns=weights_smoothed.columns)
    for dt in weights_smoothed.index:
        pos.loc[dt, :] = float(direction.reindex([dt]).iloc[0]) * weights_smoothed.loc[dt].values
    return pos

# ---------------------------
# Backtest core: turnover, costs, vol-target, metrics
# ---------------------------

def pnl_turnover_vt(returns: pd.DataFrame, pos_target: pd.DataFrame,
                    tc: float = 0.001, roll_vol_window: int = 60, vol_target: float = 0.30,
                    trading_days: int = 365, min_rebal_threshold: float = 0.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute daily raw PnL, turnover L1, net PnL after costs, vol-targeted PnL.
    Returns:
      - daily_vt: vol-targeted daily returns series
      - turnover: L1 turnover series (target-prev)
      - daily_net: daily returns after costs (before vol targeting)
    Note: pos_target are daily target positions; positions in effect for day t are pos_prev = pos_target.shift(1)
    """
    returns = returns.reindex(pos_target.index)
    pos_prev = pos_target.shift(1).fillna(0.0)
    daily_raw = (pos_prev * returns).sum(axis=1)
    turnover = (pos_target - pos_prev).abs().sum(axis=1)
    if min_rebal_threshold > 0:
        turnover = turnover.where(turnover > min_rebal_threshold, 0.0)
    costs = tc * turnover
    daily_net = daily_raw - costs
    rolling_vol = daily_net.rolling(roll_vol_window, min_periods=20).std() * math.sqrt(trading_days)
    scaling = (vol_target / (rolling_vol + 1e-9)).fillna(1.0).clip(0.05, 10.0)
    daily_vt = daily_net * scaling
    return daily_vt, turnover, daily_net

def perf_metrics(daily_series: pd.Series, trading_days: int = 365) -> dict:
    s = daily_series.dropna()
    if len(s) < 2 or s.std() == 0:
        return {"ann_return": np.nan, "ann_vol": np.nan, "sharpe": np.nan, "max_dd": np.nan}
    ann_ret = (1 + s).prod() ** (trading_days / len(s)) - 1
    ann_vol = s.std() * math.sqrt(trading_days)
    sharpe = (s.mean() * trading_days) / (s.std() * math.sqrt(trading_days) + 1e-12)
    # max drawdown on cumulative
    cum = (1 + s).cumprod()
    dd = cum.cummax() - cum
    max_dd = dd.max()
    return {"ann_return": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe, "max_dd": max_dd}

def print_metrics(metrics: dict, title: str = ""):
    if title:
        print(title)
    print(
        f"  Ann Return: {metrics['ann_return']:.2%}\n"
        f"  Ann Vol:    {metrics['ann_vol']:.2%}\n"
        f"  Sharpe:     {metrics['sharpe']:.2f}\n"
        f"  Max DD:     {metrics['max_dd']:.2%}\n"
    )


# ---------------------------
# Wrapper: run full pipeline
# ---------------------------

def run_backtest_pipeline(returns: pd.DataFrame,
                          window: int = 500, lag_T: int = 7,
                          weight_alpha: float = 0.2, ema_signal: int = 20,
                          hyst_k: float = 0.06, roll_vol_window: int = 60,
                          tc: float = 0.001, vol_target: float = 0.30,
                          reg_eps: float = 1e-8, save_dir: Optional[str] = None,
                          verbose: bool = True) -> dict:
    """
    Full pipeline: compute rolling tICA weights; factor; smooth weights; hysteresis signal; positions;
    compute pnl/turnover/vol-targeted returns; return dict with series and metrics.
    """
    if verbose: print(f"run_backtest_pipeline: window={window}, lag_T={lag_T}")
    weights = compute_rolling_tica_weights(returns, window=window, lag_T=lag_T, reg_eps=reg_eps, verbose=verbose)
    factor = compute_factor_series_from_weights(returns, weights, window=window)
    weights_sm = smooth_weights(weights, alpha=weight_alpha)
    direction = hysteresis_signal(factor, ema_span=ema_signal, hysteresis_k=hyst_k, vol_window=roll_vol_window)
    pos = build_positions(direction, weights_sm)
    daily_vt, turnover, daily_net = pnl_turnover_vt(returns, pos, tc=tc, roll_vol_window=roll_vol_window,
                                                    vol_target=vol_target, trading_days=365)
    # metrics
    metrics = perf_metrics(daily_vt)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        weights.to_parquet(os.path.join(save_dir, "tica_weights.parquet"))
        weights_sm.to_parquet(os.path.join(save_dir, "tica_weights_smoothed.parquet"))
        factor.to_frame("factor").to_parquet(os.path.join(save_dir, "factor.parquet"))
        daily_vt.to_frame("daily_vt").to_parquet(os.path.join(save_dir, "daily_vt.parquet"))
        turnover.to_frame("daily_vt").to_parquet(os.path.join(save_dir, "turnover.parquet"))
        daily_net.to_frame("daily_vt").to_parquet(os.path.join(save_dir, "daily_net.parquet"))
        pd.Series(metrics).to_frame("value").to_parquet(os.path.join(save_dir, "metrics.parquet"))
    out = {
        "weights_raw": weights,
        "weights_smoothed": weights_sm,
        "factor": factor,
        "direction": direction,
        "pos_target": pos,
        "daily_vt": daily_vt,
        "turnover": turnover,
        "daily_net": daily_net,
        "metrics": metrics
    }
    return out

# ---------------------------
# Simple plotting helpers
# ---------------------------

def plot_cum_and_turnover(daily_vt: pd.Series, turnover: pd.Series, oos_start: Optional[pd.Timestamp] = None,
                          title:str = "Cumulative / Turnover", figsize=(12,8)):
    plt.figure(figsize=figsize)
    plt.subplot(2,1,1)
    cum = (1 + daily_vt.fillna(0)).cumprod() - 1
    plt.plot(cum.index, cum.values, label="Cumulative (vol-targeted)")
    if oos_start is not None:
        plt.axvline(oos_start, color="red", linestyle="--", label="OOS start")
    plt.legend(); plt.grid(alpha=0.3)
    plt.title(title)
    plt.subplot(2,1,2)
    plt.plot(turnover.index, turnover.values, label="Turnover (L1)")
    if oos_start is not None:
        plt.axvline(oos_start, color="red", linestyle="--")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()


# ---------------------------
# Grid search (IS-only scoring) + heatmap plotting
# ---------------------------

from typing import List, Tuple, Dict
import seaborn as sns

def grid_search_tica_is(returns: pd.DataFrame,
                        window_list: List[int],
                        lag_list: List[int],
                        # pipeline params forwarded to run_backtest_pipeline
                        weight_alpha: float = 0.2,
                        ema_signal: int = 20,
                        hyst_k: float = 0.06,
                        roll_vol_window: int = 60,
                        tc: float = 0.001,
                        vol_target: float = 0.30,
                        reg_eps: float = 1e-8,
                        oos_days: int = 365,
                        max_assets: Optional[int] = None,
                        verbose: bool = True
                        ) -> Dict[str, pd.DataFrame]:
    """
    Run grid search over window_list x lag_list but compute metrics ONLY on IS (everything before last `oos_days`).
    Returns dict with DataFrames: 'return', 'sharpe', 'turnover' indexed by window and columns = lag.
    NOTE: This is IS-only scoring for parameter selection.
    """
    # optionally limit assets for speed
    returns_all = returns.copy()
    if max_assets is not None and len(returns_all.columns) > max_assets:
        returns_all = returns_all.iloc[:, :max_assets]

    dates_all = returns_all.index
    if len(dates_all) <= oos_days + 50:
        raise ValueError("Not enough history for chosen oos_days.")
    oos_start = dates_all[-oos_days]
    is_mask = dates_all < oos_start

    # prepare result frames
    results_return = pd.DataFrame(index=window_list, columns=lag_list, dtype=float)
    results_sharpe = pd.DataFrame(index=window_list, columns=lag_list, dtype=float)
    results_turnover = pd.DataFrame(index=window_list, columns=lag_list, dtype=float)

    total = len(window_list) * len(lag_list)
    i = 0
    for W in window_list:
        for L in lag_list:
            i += 1
            if verbose:
                print(f"[{i}/{total}] GRID W={W} L={L} ...", end=" ")
            try:
                out = run_backtest_pipeline(returns_all,
                                            window=W, lag_T=L,
                                            weight_alpha=weight_alpha, ema_signal=ema_signal,
                                            hyst_k=hyst_k, roll_vol_window=roll_vol_window,
                                            tc=tc, vol_target=vol_target, reg_eps=reg_eps,
                                            save_dir=None, verbose=False)
                daily_vt = out["daily_vt"]
                turnover = out["turnover"]

                # Score on IS only
                daily_vt_is = daily_vt.loc[daily_vt.index < oos_start].dropna()
                turnover_is = turnover.loc[turnover.index < oos_start].dropna()

                # compute metrics (annual return & sharpe) using existing perf_metrics
                metrics = perf_metrics(daily_vt_is)
                ann_ret = metrics["ann_return"]
                sharpe = metrics["sharpe"]
                avg_turn = float(turnover_is.mean()) if len(turnover_is)>0 else np.nan

                results_return.loc[W, L] = ann_ret
                results_sharpe.loc[W, L] = sharpe
                results_turnover.loc[W, L] = avg_turn

                if verbose:
                    print(f"IS ann_ret={ann_ret:.2%}, sharpe={sharpe:.2f}, turn={avg_turn:.3f}")
            except Exception as e:
                if verbose:
                    print("error:", str(e))
                results_return.loc[W, L] = np.nan
                results_sharpe.loc[W, L] = np.nan
                results_turnover.loc[W, L] = np.nan

    return {"return": results_return, "sharpe": results_sharpe, "turnover": results_turnover, "oos_start": oos_start}


def plot_grid_heatmaps(results: Dict[str, pd.DataFrame],
                       cmap_return: str = "RdYlGn",
                       cmap_sharpe: str = "RdYlBu_r",
                       cmap_turn: str = "viridis",
                       fmt_return: str = ".2f",
                       fmt_sharpe: str = ".2f",
                       fmt_turn: str = ".3f"):
    """
    Plot three heatmaps in a row: return(%), sharpe, turnover.
    `results` is the dict returned by grid_search_tica_is.
    """
    ret_df = results["return"]
    sharpe_df = results["sharpe"]
    turn_df = results["turnover"]

    sns.set_theme(style="white")
    plt.figure(figsize=(18,5))

    plt.subplot(1,3,1)
    sns.heatmap(ret_df.multiply(100).astype(float), annot=True, fmt=fmt_return, cmap=cmap_return,
                cbar_kws={"label":"Annual Return (%)"})
    plt.title("IS Annual Return (%)")
    plt.xlabel("Lag T"); plt.ylabel("Window")

    plt.subplot(1,3,2)
    sns.heatmap(sharpe_df.astype(float), annot=True, fmt=fmt_sharpe, cmap=cmap_sharpe,
                cbar_kws={"label":"Sharpe"})
    plt.title("IS Sharpe")
    plt.xlabel("Lag T"); plt.ylabel("")

    plt.subplot(1,3,3)
    sns.heatmap(turn_df.astype(float), annot=True, fmt=fmt_turn, cmap=cmap_turn,
                cbar_kws={"label":"Avg daily L1 turnover"})
    plt.title("IS Avg daily turnover (L1)")
    plt.xlabel("Lag T"); plt.ylabel("")

    plt.tight_layout()
    plt.show()
