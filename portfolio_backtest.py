import math
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


# =========================================================
# CONFIG
# =========================================================
INITIAL_CAPITAL = 10_000.0
START_DATE = "2018-01-01"
END_DATE = None

LONG_THRESHOLD = 0.10
CASH_THRESHOLD = -0.10

TRADING_DAYS_ANNUAL = 365
RISK_FREE_RATE = 0.0
FEE_PER_TURNOVER = 0.0
MIN_STD = 1e-12

ASSETS = ["BTC", "ETH", "SOL", "DOGE", "SUI", "SPX", "BNB", "TRX", "HYPE"]
BENCHMARK_ASSET = "BTC"

INVVOL_WINDOW = 365
MIN_VOL_LOOKBACK = 90

CORR_THRESHOLD = 0.7
SHARPE_GAP = 0.25

SUBPERIOD_START = "2023-01-01"
SUBPERIOD_START_2 = "2025-08-01"

MAX_WEIGHT_CAPS = pd.Series({
    "BTC": np.inf,
    "ETH": np.inf,
    "SOL": np.inf,
    "DOGE": 0.30,
    "SUI": 0.25,
    "SPX": 0.10,
    "BNB": 0.70,
    "TRX": 0.50,
    "HYPE": 0.60,
}).reindex(ASSETS)

TPI_FILES = {
    "TOTAL": "TOTAL_TPI.csv",
    "BTC": "BTC_TPI.csv",
    "ETH": "ETH_TPI.csv",
    "SOL": "SOL_TPI.csv",
    "DOGE": "DOGECOIN_TPI.csv",
    "SUI": "SUI_TPI.csv",
    "SPX": "SPX_TPI.csv",
    "BNB": "BNB_TPI.csv",
    "TRX": "TRX_TPI.csv",
    "HYPE": "HYPE_TPI.csv",
}

YF_TICKERS = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "SOL": "SOL-USD",
    "DOGE": "DOGE-USD",
    "SUI": "SUI20947-USD",
    "BNB": "BNB-USD",
    "TRX": "TRX-USD",
    "HYPE": "HYPE32196-USD",
}

LOCAL_PRICE_FILES = {
    "SPX": "SPX_PRICE.csv",
}


# =========================================================
# PATHS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "out"
PNG_DIR = OUT_DIR / "pngs"
CSV_DIR = OUT_DIR / "csvs"

PNG_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)

EQUITY_PNG = PNG_DIR / "equity_compare.png"
EQUITY_2023_PNG = PNG_DIR / "equity_compare_2023_onwards.png"
EQUITY_2025_08_PNG = PNG_DIR / "equity_compare_2025_08_01_onwards.png"

METRICS_PNG = PNG_DIR / "metrics_compare.png"
METRICS_2023_PNG = PNG_DIR / "metrics_compare_2023_onwards.png"

AVG_WEIGHTS_PNG = PNG_DIR / "average_active_weights_table.png"

EQUITY_CSV = CSV_DIR / "equity_curves.csv"
EQUITY_2023_CSV = CSV_DIR / "equity_curves_2023_onwards.csv"
EQUITY_2025_08_CSV = CSV_DIR / "equity_curves_2025_08_01_onwards.csv"

METRICS_CSV = CSV_DIR / "metrics_compare.csv"
METRICS_2023_CSV = CSV_DIR / "metrics_compare_2023_onwards.csv"

AVG_WEIGHTS_CSV = CSV_DIR / "average_active_weights.csv"

INVVOL_WEIGHTS_CSV = CSV_DIR / "weights_invvol_365d_capped.csv"
INVVOL_TRADES_CSV = CSV_DIR / "trades_invvol_365d_capped.csv"

CORR180_WEIGHTS_CSV = CSV_DIR / "weights_corr_filter_180d_capped.csv"
CORR180_TRADES_CSV = CSV_DIR / "trades_corr_filter_180d_capped.csv"

CORR180_FULL180_WEIGHTS_CSV = CSV_DIR / "weights_corr_filter_180d_full_180d_capped.csv"
CORR180_FULL180_TRADES_CSV = CSV_DIR / "trades_corr_filter_180d_full_180d_capped.csv"

BUYHOLD_TRADES_CSV = CSV_DIR / "trades_buy_hold_btc.csv"


# =========================================================
# HELPERS
# =========================================================
def safe_to_csv(df, path, index=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)


def require_file(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Required file not found: {p}")
    return p


def round2_or_nan(x):
    return np.nan if pd.isna(x) else round(float(x), 2)


def normalize_to_one(series):
    clean = series.dropna()
    if clean.empty:
        return series.copy()
    return series / clean.iloc[0]


def normalize_from_date(series, start_date):
    s = series.dropna().copy()
    s = s.loc[s.index >= pd.Timestamp(start_date)]
    if s.empty:
        return s
    return s / s.iloc[0]


def first_row_of_month(index):
    s = pd.Series(index.to_period("M"), index=index)
    return ~s.duplicated()


# =========================================================
# LOADING
# =========================================================
def load_two_col_csv(path, out_col, value_candidates):
    path = require_file(path)
    df = pd.read_csv(path)

    cols = {c.lower().strip(): c for c in df.columns}

    date_col = next((cols[c] for c in ["time", "date", "datetime", "timestamp"] if c in cols), None)
    if date_col is None:
        raise ValueError(f"{path}: missing date column")

    value_col = next((cols[c] for c in value_candidates if c in cols), None)
    if value_col is None:
        raise ValueError(f"{path}: missing value column. Expected one of {value_candidates}")

    out = df[[date_col, value_col]].copy()
    out.columns = ["time", out_col]
    out["time"] = pd.to_datetime(out["time"], dayfirst=True, errors="coerce")
    out[out_col] = out[out_col].astype(str).str.replace(",", ".", regex=False)
    out[out_col] = pd.to_numeric(out[out_col], errors="coerce")
    out = out.dropna(subset=["time"]).drop_duplicates(subset=["time"]).sort_values("time").set_index("time")

    if START_DATE:
        out = out.loc[out.index >= pd.Timestamp(START_DATE)]
    if END_DATE:
        out = out.loc[out.index <= pd.Timestamp(END_DATE)]

    if out.empty:
        raise ValueError(f"{path}: no valid rows after parsing")

    return out


def load_tpi_csv(path, name):
    return load_two_col_csv(path, f"{name}_TPI", ["tpi", "value", "signal", "close"])


def load_price_csv(path, asset):
    return load_two_col_csv(path, asset, ["close", "adj close", "price", "value"])


def safe_download_yf_close(ticker, start=None, end=None):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"Yahoo Finance returned no data for {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        if ("Close", ticker) in df.columns:
            out = df[("Close", ticker)].to_frame(name="Close")
        elif "Close" in df.columns.get_level_values(0):
            out = df.xs("Close", axis=1, level=0).iloc[:, [0]].rename(columns=lambda _: "Close")
        else:
            raise RuntimeError(f"No Close column for {ticker}")
    else:
        if "Close" not in df.columns:
            raise RuntimeError(f"No Close column for {ticker}")
        out = df[["Close"]].copy()

    out.index = pd.to_datetime(out.index)
    return out.sort_index()


def load_all_tpis(base_dir):
    return {name: load_tpi_csv(base_dir / fn, name).iloc[:, 0] for name, fn in TPI_FILES.items()}


def load_all_prices(base_dir):
    price_frames = {}

    for asset in ASSETS:
        if asset in LOCAL_PRICE_FILES:
            s = load_price_csv(base_dir / LOCAL_PRICE_FILES[asset], asset).iloc[:, 0]
        else:
            yf_df = safe_download_yf_close(YF_TICKERS[asset], start=START_DATE, end=END_DATE)
            s = yf_df["Close"].rename(asset)
        price_frames[asset] = s

    prices = pd.concat(price_frames.values(), axis=1).sort_index()

    if START_DATE:
        prices = prices.loc[prices.index >= pd.Timestamp(START_DATE)]
    if END_DATE:
        prices = prices.loc[prices.index <= pd.Timestamp(END_DATE)]

    if prices.empty:
        raise RuntimeError("Price dataframe is empty")

    return prices


def build_master_dataframe(tpis, prices):
    df = pd.concat([prices] + [s.to_frame(name=s.name) for s in tpis.values()], axis=1).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.loc[df.index.notna()]

    if START_DATE:
        df = df.loc[df.index >= pd.Timestamp(START_DATE)]
    if END_DATE:
        df = df.loc[df.index <= pd.Timestamp(END_DATE)]

    if df.empty:
        raise RuntimeError("Combined master dataframe is empty")

    return df


# =========================================================
# SIGNALS
# =========================================================
def hysteresis_signal_state(series, long_threshold=LONG_THRESHOLD, cash_threshold=CASH_THRESHOLD, initial_state=False):
    s = series.copy().sort_index()
    state = pd.Series(np.nan, index=s.index, dtype=float)

    state[s >= long_threshold] = 1.0
    state[s <= cash_threshold] = 0.0

    if len(state) > 0 and pd.isna(state.iloc[0]):
        state.iloc[0] = 1.0 if initial_state else 0.0

    return state.ffill().fillna(0.0).astype(bool)


def build_eligibility_table(returns, total_tpi, asset_tpis):
    idx = returns.index
    assets = list(returns.columns)

    total_long = hysteresis_signal_state(total_tpi.reindex(idx), initial_state=False)

    eligible = pd.DataFrame(False, index=idx, columns=assets)
    for a in assets:
        asset_sig = hysteresis_signal_state(asset_tpis[a].reindex(idx), initial_state=False)
        eligible[a] = total_long & asset_sig

    live = returns.notna()
    return total_long, eligible, live


# =========================================================
# SCORE ENGINE
# =========================================================
def rolling_inv_vol_scores(returns, window, min_periods=MIN_VOL_LOOKBACK):
    scores = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
    for col in returns.columns:
        vol = returns[col].rolling(window=window, min_periods=min_periods).std(ddof=1)
        scores[col] = (1.0 / vol).replace([np.inf, -np.inf], np.nan)
    return scores


def rolling_sharpe_scores(returns, window, min_periods=MIN_VOL_LOOKBACK):
    rolling_mean = returns.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = returns.rolling(window=window, min_periods=min_periods).std(ddof=1)
    sharpe = rolling_mean / rolling_std.replace(0.0, np.nan)
    sharpe = sharpe.replace([np.inf, -np.inf], np.nan)
    return sharpe


def rolling_vol_scores(returns, window, min_periods=MIN_VOL_LOOKBACK):
    return returns.rolling(window=window, min_periods=min_periods).std(ddof=1)


# =========================================================
# CORRELATION FILTER
# =========================================================
def filter_correlated_assets_by_sharpe_and_vol(
    eligible_assets,
    corr_matrix,
    sharpe_row,
    vol_row,
    threshold,
    sharpe_gap=0.25
):
    selected = list(eligible_assets)
    if len(selected) <= 1:
        return selected

    changed = True
    while changed:
        changed = False

        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                a = selected[i]
                b = selected[j]

                corr_ab = corr_matrix.loc[a, b]
                if pd.isna(corr_ab):
                    continue

                if corr_ab > threshold:
                    sharpe_a = sharpe_row.get(a, np.nan)
                    sharpe_b = sharpe_row.get(b, np.nan)
                    vol_a = vol_row.get(a, np.nan)
                    vol_b = vol_row.get(b, np.nan)

                    if pd.notna(sharpe_a) and pd.notna(sharpe_b):
                        if abs(sharpe_a - sharpe_b) >= sharpe_gap:
                            loser = b if sharpe_a >= sharpe_b else a
                        else:
                            loser = a if vol_a >= vol_b else b
                    elif pd.notna(sharpe_a) and pd.isna(sharpe_b):
                        loser = b
                    elif pd.isna(sharpe_a) and pd.notna(sharpe_b):
                        loser = a
                    else:
                        if pd.notna(vol_a) and pd.notna(vol_b):
                            loser = a if vol_a >= vol_b else b
                        elif pd.notna(vol_a):
                            loser = a
                        elif pd.notna(vol_b):
                            loser = b
                        else:
                            loser = b

                    selected.remove(loser)
                    changed = True
                    break

            if changed:
                break

    return selected


# =========================================================
# CAP LOGIC
# =========================================================
def capped_proportional_weights(score_row, eligible_row, live_row, caps, tol=1e-12):
    allowed = eligible_row.astype(bool) & live_row.astype(bool)

    scores = score_row.copy()
    scores[~allowed] = np.nan
    scores = scores.clip(lower=0.0).fillna(0.0)

    if scores.sum() <= tol:
        return pd.Series(0.0, index=score_row.index, dtype=float)

    weights = pd.Series(0.0, index=score_row.index, dtype=float)
    remaining_budget = 1.0

    while remaining_budget > tol:
        remaining_assets = [a for a in scores.index if scores[a] > 0 and weights[a] < caps.get(a, np.inf) - tol]
        if not remaining_assets:
            break

        sub_scores = scores.loc[remaining_assets]
        if sub_scores.sum() <= tol:
            break

        proposal = remaining_budget * (sub_scores / sub_scores.sum())
        allocated_this_round = 0.0

        for a in remaining_assets:
            cap_a = caps.get(a, np.inf)
            room = max(cap_a - weights[a], 0.0)
            alloc = min(proposal[a], room)
            weights[a] += alloc
            allocated_this_round += alloc

        if allocated_this_round <= tol:
            break

        remaining_budget = 1.0 - weights.sum()

    return weights.clip(lower=0.0)


# =========================================================
# WEIGHT BUILDING
# =========================================================
def signals_materially_changed(current_eligible, previous_eligible):
    if previous_eligible is None:
        return True
    return not current_eligible.fillna(False).equals(previous_eligible.fillna(False))


def rebuild_capped_weights(dt, eligible_today, live_today, scores_row, returns, corr_window, caps, sharpe_scores, vol_scores):
    assets = list(eligible_today.index)
    candidates = [a for a in assets if bool(eligible_today[a]) and bool(live_today[a])]

    if len(candidates) == 0:
        return pd.Series(0.0, index=assets, dtype=float)

    if len(candidates) == 1:
        filtered_eligible = pd.Series(False, index=assets)
        filtered_eligible.loc[candidates] = True
        return capped_proportional_weights(scores_row, filtered_eligible, live_today, caps)

    window_returns = returns.loc[:dt, candidates].tail(corr_window)
    corr_matrix = window_returns.corr(min_periods=MIN_VOL_LOOKBACK)

    survivors = filter_correlated_assets_by_sharpe_and_vol(
        eligible_assets=candidates,
        corr_matrix=corr_matrix,
        sharpe_row=sharpe_scores.loc[dt],
        vol_row=vol_scores.loc[dt],
        threshold=CORR_THRESHOLD,
        sharpe_gap=SHARPE_GAP
    )

    filtered_eligible = pd.Series(False, index=assets)
    filtered_eligible.loc[survivors] = True

    return capped_proportional_weights(scores_row, filtered_eligible, live_today, caps)


def build_tpi_sleeve_from_scores(returns, total_tpi, asset_tpis, scores, caps):
    idx = returns.index
    assets = list(returns.columns)

    total_long, eligible, live = build_eligibility_table(returns, total_tpi, asset_tpis)
    rebalance_month = first_row_of_month(idx)

    weights = pd.DataFrame(0.0, index=idx, columns=assets)
    prev_eligible = None
    prev_w = pd.Series(0.0, index=assets)

    for dt in idx:
        eligible_today = eligible.loc[dt]
        live_today = live.loc[dt]
        total_is_long = bool(total_long.loc[dt])

        force_rebalance = (
            dt == idx[0]
            or rebalance_month.loc[dt]
            or signals_materially_changed(eligible_today, prev_eligible)
        )

        if not total_is_long:
            curr_w = pd.Series(0.0, index=assets)
        else:
            if force_rebalance:
                curr_w = capped_proportional_weights(scores.loc[dt], eligible_today, live_today, caps)
            else:
                curr_w = prev_w.copy()
                curr_w = curr_w.where(eligible_today & live_today, 0.0)
                curr_w = curr_w.where(curr_w > 0, 0.0)

                if curr_w.sum() == 0:
                    curr_w = capped_proportional_weights(scores.loc[dt], eligible_today, live_today, caps)

        weights.loc[dt] = curr_w.values
        prev_w = curr_w
        prev_eligible = eligible_today.copy()

    return weights


def build_tpi_sleeve_with_corr_filter(returns, total_tpi, asset_tpis, invvol_scores, corr_window, caps, tie_vol_window):
    idx = returns.index
    assets = list(returns.columns)

    total_long, eligible, live = build_eligibility_table(returns, total_tpi, asset_tpis)
    rebalance_month = first_row_of_month(idx)

    sharpe_scores = rolling_sharpe_scores(returns, corr_window, min_periods=MIN_VOL_LOOKBACK)
    vol_scores = rolling_vol_scores(returns, tie_vol_window, min_periods=MIN_VOL_LOOKBACK)

    weights = pd.DataFrame(0.0, index=idx, columns=assets)
    prev_eligible = None
    prev_w = pd.Series(0.0, index=assets)

    for dt in idx:
        eligible_today = eligible.loc[dt]
        live_today = live.loc[dt]
        total_is_long = bool(total_long.loc[dt])

        force_rebalance = (
            dt == idx[0]
            or rebalance_month.loc[dt]
            or signals_materially_changed(eligible_today, prev_eligible)
        )

        if not total_is_long:
            curr_w = pd.Series(0.0, index=assets)
        else:
            if force_rebalance:
                curr_w = rebuild_capped_weights(
                    dt=dt,
                    eligible_today=eligible_today,
                    live_today=live_today,
                    scores_row=invvol_scores.loc[dt],
                    returns=returns,
                    corr_window=corr_window,
                    caps=caps,
                    sharpe_scores=sharpe_scores,
                    vol_scores=vol_scores
                )
            else:
                curr_w = prev_w.copy()
                curr_w = curr_w.where(eligible_today & live_today, 0.0)
                curr_w = curr_w.where(curr_w > 0, 0.0)

                if curr_w.sum() == 0:
                    curr_w = rebuild_capped_weights(
                        dt=dt,
                        eligible_today=eligible_today,
                        live_today=live_today,
                        scores_row=invvol_scores.loc[dt],
                        returns=returns,
                        corr_window=corr_window,
                        caps=caps,
                        sharpe_scores=sharpe_scores,
                        vol_scores=vol_scores
                    )

        weights.loc[dt] = curr_w.values
        prev_w = curr_w
        prev_eligible = eligible_today.copy()

    return weights


# =========================================================
# EQUITY / TRADES
# =========================================================
def turnover_from_weights(weights):
    return weights.fillna(0.0).diff().abs().sum(axis=1).fillna(weights.iloc[0].abs().sum())


def compute_equity_from_weights(returns, weights):
    aligned_r = returns.reindex(weights.index)
    effective_w = weights.shift(1).fillna(0.0)
    gross_ret = (effective_w * aligned_r.fillna(0.0)).sum(axis=1)
    net_ret = gross_ret - turnover_from_weights(weights) * FEE_PER_TURNOVER
    equity = INITIAL_CAPITAL * (1.0 + net_ret.fillna(0.0)).cumprod()
    return net_ret, equity


def compute_buy_hold_equity_from_price(price_series):
    p = price_series.dropna().copy()
    if p.empty:
        raise RuntimeError("Benchmark price series is empty")
    equity = INITIAL_CAPITAL * (p / p.iloc[0])
    daily_ret = p.pct_change().fillna(0.0)
    return daily_ret, equity


def extract_total_tpi_cycle_trades(equity, total_long_series):
    eq = equity.reindex(total_long_series.index).ffill()
    tl = total_long_series.copy()

    trades = []
    in_trade = False
    entry_dt = None
    entry_eq = None

    for dt in tl.index:
        is_long = bool(tl.loc[dt])

        if not in_trade and is_long:
            in_trade = True
            entry_dt = dt
            entry_eq = float(eq.loc[dt])

        elif in_trade and not is_long:
            exit_eq = float(eq.loc[dt])
            profit_dollar = exit_eq - entry_eq
            trades.append({
                "entry_date": entry_dt,
                "exit_date": dt,
                "entry_equity": entry_eq,
                "exit_equity": exit_eq,
                "profit_dollar": profit_dollar,
                "trade_return": exit_eq / entry_eq - 1.0 if entry_eq else np.nan,
                "days": (dt - entry_dt).days,
                "win": bool(profit_dollar > 0),
                "status": "closed",
            })
            in_trade = False
            entry_dt = None
            entry_eq = None

    if in_trade:
        dt = tl.index[-1]
        exit_eq = float(eq.loc[dt])
        profit_dollar = exit_eq - entry_eq
        trades.append({
            "entry_date": entry_dt,
            "exit_date": dt,
            "entry_equity": entry_eq,
            "exit_equity": exit_eq,
            "profit_dollar": profit_dollar,
            "trade_return": exit_eq / entry_eq - 1.0 if entry_eq else np.nan,
            "days": (dt - entry_dt).days,
            "win": bool(profit_dollar > 0),
            "status": "open",
        })

    return pd.DataFrame(trades)


def extract_single_buy_hold_trade(price_series):
    p = price_series.dropna()
    if len(p) < 2:
        return pd.DataFrame(columns=[
            "entry_date", "exit_date", "entry_equity", "exit_equity",
            "profit_dollar", "trade_return", "days", "win", "status"
        ])

    exit_eq = INITIAL_CAPITAL * float(p.iloc[-1]) / float(p.iloc[0])
    return pd.DataFrame([{
        "entry_date": p.index[0],
        "exit_date": p.index[-1],
        "entry_equity": INITIAL_CAPITAL,
        "exit_equity": exit_eq,
        "profit_dollar": exit_eq - INITIAL_CAPITAL,
        "trade_return": exit_eq / INITIAL_CAPITAL - 1.0,
        "days": (p.index[-1] - p.index[0]).days,
        "win": bool(exit_eq > INITIAL_CAPITAL),
        "status": "closed",
    }])


# =========================================================
# METRICS
# =========================================================
def max_drawdown_pct(equity):
    eq = equity.dropna()
    if eq.empty:
        return np.nan
    running_peak = eq.cummax()
    return ((eq - running_peak) / running_peak * 100.0).min()


def annualized_std_pct(daily_returns):
    r = daily_returns.dropna()
    if len(r) < 2:
        return np.nan
    std = r.std(ddof=1)
    if pd.isna(std):
        return np.nan
    return std * math.sqrt(TRADING_DAYS_ANNUAL) * 100.0


def cagr_pct(equity):
    eq = equity.dropna()
    if len(eq) < 2:
        return np.nan

    start_val = float(eq.iloc[0])
    end_val = float(eq.iloc[-1])
    if start_val <= 0 or end_val <= 0:
        return np.nan

    elapsed_days = (eq.index[-1] - eq.index[0]).days
    if elapsed_days <= 0:
        return np.nan

    years = elapsed_days / TRADING_DAYS_ANNUAL
    if years <= 0:
        return np.nan

    return ((end_val / start_val) ** (1.0 / years) - 1.0) * 100.0


def sharpe_ratio(daily_returns):
    r = daily_returns.dropna()
    if len(r) < 2:
        return np.nan
    std = r.std(ddof=1)
    if pd.isna(std) or std < MIN_STD:
        return np.nan
    mean_excess = r.mean() - (RISK_FREE_RATE / TRADING_DAYS_ANNUAL)
    return mean_excess / std * math.sqrt(TRADING_DAYS_ANNUAL)


def sortino_ratio(daily_returns):
    r = daily_returns.dropna()
    if len(r) < 2:
        return np.nan
    downside = np.minimum(r, 0.0)
    downside_dev = math.sqrt(np.mean(np.square(downside)))
    if downside_dev < MIN_STD:
        return np.nan
    mean_excess = r.mean() - (RISK_FREE_RATE / TRADING_DAYS_ANNUAL)
    return mean_excess / downside_dev * math.sqrt(TRADING_DAYS_ANNUAL)


def omega_ratio(daily_returns, threshold=0.0):
    r = daily_returns.dropna()
    if len(r) == 0:
        return np.nan

    gains = np.maximum(r - threshold, 0.0).sum()
    losses = np.maximum(threshold - r, 0.0).sum()

    if losses > 0:
        return gains / losses
    if gains > 0:
        return np.inf
    return np.nan


def profit_factor(trades):
    if trades.empty:
        return np.nan
    closed = trades[trades["status"] == "closed"] if "status" in trades.columns else trades
    gross_profit = closed.loc[closed["profit_dollar"] > 0, "profit_dollar"].sum()
    gross_loss = -closed.loc[closed["profit_dollar"] < 0, "profit_dollar"].sum()

    if gross_loss > 0:
        return gross_profit / gross_loss
    if gross_profit > 0:
        return np.inf
    return np.nan


def win_rate_pct(trades):
    if trades.empty:
        return np.nan
    closed = trades[trades["status"] == "closed"] if "status" in trades.columns else trades
    if len(closed) == 0:
        return np.nan
    return (closed["profit_dollar"] > 0).mean() * 100.0


def build_metrics(name, equity, daily_returns, trades):
    closed_count = int(len(trades[trades["status"] == "closed"])) if "status" in trades.columns else int(len(trades))
    return {
        "Portfolio": name,
        "CAGR %": round2_or_nan(cagr_pct(equity)),
        "Std Dev %": round2_or_nan(annualized_std_pct(daily_returns)),
        "Max Equity Drawdown %": round2_or_nan(max_drawdown_pct(equity)),
        "Sharpe Ratio": round2_or_nan(sharpe_ratio(daily_returns)),
        "Sortino Ratio": round2_or_nan(sortino_ratio(daily_returns)),
        "Omega Ratio": round2_or_nan(omega_ratio(daily_returns, threshold=0.0)),
        "Profit Factor": round2_or_nan(profit_factor(trades)),
        "Win Rate %": round2_or_nan(win_rate_pct(trades)),
        "Number of Trades": closed_count,
    }


# =========================================================
# AVERAGE ACTIVE WEIGHTS
# =========================================================
def average_active_weights(weights):
    active_mask = weights.sum(axis=1) > 0
    if active_mask.sum() == 0:
        return pd.Series(0.0, index=weights.columns)
    return weights.loc[active_mask].mean()


# =========================================================
# VISUALS
# =========================================================
def export_table_png(df, path, caption, percent_cols=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    percent_cols = set(percent_cols or [])
    export_df = df.copy()

    for col in export_df.columns:
        if pd.api.types.is_numeric_dtype(export_df[col]):
            if col in percent_cols:
                export_df[col] = export_df[col].apply(lambda x: "" if pd.isna(x) else f"{x:.2%}")
            else:
                export_df[col] = export_df[col].apply(lambda x: "" if pd.isna(x) else f"{x:,.2f}")

    nrows, ncols = export_df.shape
    fig_width = max(8, min(24, 0.9 * ncols + 4))
    fig_height = max(2.2, min(18, 0.42 * (nrows + 1) + 0.9))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.set_title(caption, fontsize=13, fontweight="bold", pad=10)

    table = ax.table(
        cellText=export_df.values,
        colLabels=export_df.columns,
        loc="center",
        cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.12)

    try:
        table.auto_set_column_width(col=list(range(ncols)))
    except Exception:
        pass

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d1d5db")
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#1f2937")
            cell.PAD = 0.015
        else:
            cell.set_facecolor("#f8fafc" if row % 2 == 0 else "white")
            cell.PAD = 0.01

    plt.tight_layout(pad=0.8)
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_equity_curves(curves_df, path, title):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(13, 7))
    for col in curves_df.columns:
        plt.plot(curves_df.index, curves_df[col], label=col, linewidth=2)

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


# =========================================================
# SUBPERIOD HELPERS
# =========================================================
def build_subperiod_metrics(portfolio_results, start_date):
    sub_metrics = []

    for p in portfolio_results:
        eq = p["equity"].dropna().loc[lambda x: x.index >= pd.Timestamp(start_date)]
        dr = p["daily_ret"].dropna().loc[lambda x: x.index >= pd.Timestamp(start_date)]

        if p["name"] != f"Buy & Hold {BENCHMARK_ASSET}":
            trades_sub = p["trades"].copy()
            if not trades_sub.empty and "entry_date" in trades_sub.columns:
                trades_sub["entry_date"] = pd.to_datetime(trades_sub["entry_date"])
                trades_sub["exit_date"] = pd.to_datetime(trades_sub["exit_date"])
                trades_sub = trades_sub[trades_sub["entry_date"] >= pd.Timestamp(start_date)].copy()
        else:
            trades_sub = p["trades"].copy()

        sub_metrics.append(build_metrics(p["name"], eq, dr, trades_sub))

    return pd.DataFrame(sub_metrics)


# =========================================================
# MAIN
# =========================================================
def main():
    tpis = load_all_tpis(BASE_DIR)
    prices = load_all_prices(BASE_DIR)
    master = build_master_dataframe(tpis, prices)

    print("Backtest start:", master.index.min())
    print("Backtest end:", master.index.max())

    prices = master[ASSETS].copy().sort_index()
    returns = prices.pct_change().fillna(0.0)

    total_tpi = master["TOTAL_TPI"].copy()
    asset_tpis = {a: master[f"{a}_TPI"].copy() for a in ASSETS}
    total_long = hysteresis_signal_state(total_tpi.reindex(returns.index), initial_state=False)

    portfolio_results = []

    scores_invvol_365 = rolling_inv_vol_scores(
        returns,
        365,
        min_periods=MIN_VOL_LOOKBACK
    )

    scores_invvol_180 = rolling_inv_vol_scores(
        returns,
        180,
        min_periods=MIN_VOL_LOOKBACK
    )

    # 1) TPI Sleeve Inv Vol 1Y Capped
    w_invvol = build_tpi_sleeve_from_scores(
        returns=returns,
        total_tpi=total_tpi,
        asset_tpis=asset_tpis,
        scores=scores_invvol_365,
        caps=MAX_WEIGHT_CAPS
    )
    r_invvol, e_invvol = compute_equity_from_weights(returns, w_invvol)
    t_invvol = extract_total_tpi_cycle_trades(e_invvol, total_long)
    safe_to_csv(w_invvol, INVVOL_WEIGHTS_CSV, index=True)
    safe_to_csv(t_invvol, INVVOL_TRADES_CSV, index=False)
    portfolio_results.append({
        "name": "TPI Sleeve (Inv Vol 1Y Capped)",
        "weights": w_invvol,
        "daily_ret": r_invvol,
        "equity": e_invvol,
        "trades": t_invvol,
    })

    # 2) TPI Sleeve Corr Filter 180D Capped
    #    Uses 365d inverse-vol starting weights, 180d correlation, 180d Sharpe, 365d vol tiebreak
    w_corr180 = build_tpi_sleeve_with_corr_filter(
        returns=returns,
        total_tpi=total_tpi,
        asset_tpis=asset_tpis,
        invvol_scores=scores_invvol_365,
        corr_window=180,
        caps=MAX_WEIGHT_CAPS,
        tie_vol_window=365
    )
    r_corr180, e_corr180 = compute_equity_from_weights(returns, w_corr180)
    t_corr180 = extract_total_tpi_cycle_trades(e_corr180, total_long)
    safe_to_csv(w_corr180, CORR180_WEIGHTS_CSV, index=True)
    safe_to_csv(t_corr180, CORR180_TRADES_CSV, index=False)
    portfolio_results.append({
        "name": "TPI Sleeve (Corr Filter 180D Capped)",
        "weights": w_corr180,
        "daily_ret": r_corr180,
        "equity": e_corr180,
        "trades": t_corr180,
    })

    # 3) TPI Sleeve Corr Filter 180D Full 180D Capped
    #    Uses 180d inverse-vol starting weights, 180d correlation, 180d Sharpe, 180d vol tiebreak
    w_corr180_full180 = build_tpi_sleeve_with_corr_filter(
        returns=returns,
        total_tpi=total_tpi,
        asset_tpis=asset_tpis,
        invvol_scores=scores_invvol_180,
        corr_window=180,
        caps=MAX_WEIGHT_CAPS,
        tie_vol_window=180
    )
    r_corr180_full180, e_corr180_full180 = compute_equity_from_weights(returns, w_corr180_full180)
    t_corr180_full180 = extract_total_tpi_cycle_trades(e_corr180_full180, total_long)
    safe_to_csv(w_corr180_full180, CORR180_FULL180_WEIGHTS_CSV, index=True)
    safe_to_csv(t_corr180_full180, CORR180_FULL180_TRADES_CSV, index=False)
    portfolio_results.append({
        "name": "TPI Sleeve (Corr Filter 180D Full 180D Capped)",
        "weights": w_corr180_full180,
        "daily_ret": r_corr180_full180,
        "equity": e_corr180_full180,
        "trades": t_corr180_full180,
    })

    # 4) Buy & Hold BTC
    bench_price = prices[BENCHMARK_ASSET].dropna().copy()
    bench_daily_ret, bench_equity = compute_buy_hold_equity_from_price(bench_price)
    bench_weights = pd.DataFrame(0.0, index=bench_equity.index, columns=ASSETS)
    bench_weights[BENCHMARK_ASSET] = 1.0
    bench_trades = extract_single_buy_hold_trade(bench_price)
    safe_to_csv(bench_trades, BUYHOLD_TRADES_CSV, index=False)
    portfolio_results.append({
        "name": f"Buy & Hold {BENCHMARK_ASSET}",
        "weights": bench_weights,
        "daily_ret": bench_daily_ret,
        "equity": bench_equity,
        "trades": bench_trades,
    })

    metrics_df = pd.DataFrame([
        build_metrics(p["name"], p["equity"], p["daily_ret"], p["trades"])
        for p in portfolio_results
    ])
    safe_to_csv(metrics_df, METRICS_CSV, index=False)
    export_table_png(metrics_df, METRICS_PNG, "Portfolio Metrics Comparison")

    equity_df = pd.concat(
        [normalize_to_one(p["equity"]).rename(p["name"]) for p in portfolio_results],
        axis=1
    ).dropna(how="all")
    safe_to_csv(equity_df, EQUITY_CSV, index=True)
    plot_equity_curves(equity_df, EQUITY_PNG, "Portfolio Equity Comparison")

    avg_active_weights_df = pd.DataFrame(
        [average_active_weights(p["weights"]).rename(p["name"]) for p in portfolio_results if p["name"] != f"Buy & Hold {BENCHMARK_ASSET}"]
    ).fillna(0.0)
    avg_active_weights_df.index.name = "Portfolio"
    safe_to_csv(avg_active_weights_df, AVG_WEIGHTS_CSV, index=True)

    export_table_png(
        avg_active_weights_df.reset_index(),
        AVG_WEIGHTS_PNG,
        "Average Active Weights",
        percent_cols=set(ASSETS)
    )

    equity_2023_df = pd.concat(
        [normalize_from_date(p["equity"], SUBPERIOD_START).rename(p["name"]) for p in portfolio_results],
        axis=1
    ).dropna(how="all")
    safe_to_csv(equity_2023_df, EQUITY_2023_CSV, index=True)
    plot_equity_curves(
        equity_2023_df,
        EQUITY_2023_PNG,
        f"Portfolio Equity Comparison ({SUBPERIOD_START} onwards)"
    )

    equity_2025_08_df = pd.concat(
        [normalize_from_date(p["equity"], SUBPERIOD_START_2).rename(p["name"]) for p in portfolio_results],
        axis=1
    ).dropna(how="all")
    safe_to_csv(equity_2025_08_df, EQUITY_2025_08_CSV, index=True)
    plot_equity_curves(
        equity_2025_08_df,
        EQUITY_2025_08_PNG,
        f"Portfolio Equity Comparison ({SUBPERIOD_START_2} onwards)"
    )

    metrics_2023_df = build_subperiod_metrics(portfolio_results, SUBPERIOD_START)
    safe_to_csv(metrics_2023_df, METRICS_2023_CSV, index=False)
    export_table_png(
        metrics_2023_df,
        METRICS_2023_PNG,
        f"Portfolio Metrics Comparison ({SUBPERIOD_START} onwards)"
    )

    print("\nMetrics:")
    print(metrics_df.to_string(index=False))

    print("\nMetrics 2023 onwards:")
    print(metrics_2023_df.to_string(index=False))

    print("\nAverage Active Weights:")
    print(avg_active_weights_df.to_string())

    print("\nSaved files:")
    for p in [
        EQUITY_PNG,
        EQUITY_2023_PNG,
        EQUITY_2025_08_PNG,
        METRICS_PNG,
        METRICS_2023_PNG,
        AVG_WEIGHTS_PNG,
        EQUITY_CSV,
        EQUITY_2023_CSV,
        EQUITY_2025_08_CSV,
        METRICS_CSV,
        METRICS_2023_CSV,
        AVG_WEIGHTS_CSV,
        INVVOL_WEIGHTS_CSV,
        INVVOL_TRADES_CSV,
        CORR180_WEIGHTS_CSV,
        CORR180_TRADES_CSV,
        CORR180_FULL180_WEIGHTS_CSV,
        CORR180_FULL180_TRADES_CSV,
        BUYHOLD_TRADES_CSV,
    ]:
        print(p)


if __name__ == "__main__":
    main()