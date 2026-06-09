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

TOTAL_THRESHOLD = 0.10
TRADING_DAYS_ANNUAL = 365
RISK_FREE_RATE = 0.0
FEE_PER_TURNOVER = 0.0
MIN_STD = 1e-12

ASSETS = ["BTC", "SOL", "DOGE", "SUI", "SPX", "BNB"]
BENCHMARK_ASSET = "BTC"

STANDARD_TARGET_WEIGHTS = pd.Series({
    "BTC": 0.40,
    "SOL": 0.20,
    "BNB": 0.10,
    "DOGE": 0.10,
    "SUI": 0.10,
    "SPX": 0.10,
}).reindex(ASSETS).fillna(0.0)

TPI_FILES = {
    "TOTAL": "TOTAL_TPI.csv",
    "BTC": "BTC_TPI.csv",
    "SOL": "SOL_TPI.csv",
    "DOGE": "DOGECOIN_TPI.csv",
    "SUI": "SUI_TPI.csv",
    "SPX": "SPX_TPI.csv",
    "BNB": "BNB_TPI.csv",
}

YF_TICKERS = {
    "BTC": "BTC-USD",
    "SOL": "SOL-USD",
    "DOGE": "DOGE-USD",
    "SUI": "SUI20947-USD",
    "BNB": "BNB-USD",
}

LOCAL_PRICE_FILES = {
    "SPX": "SPX_PRICE.csv",
}


# =========================================================
# PATHS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "tpi_sleeve_4portfolios"
PNG_DIR = OUT_DIR / "pngs"
CSV_DIR = OUT_DIR / "csvs"

PNG_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)

EQUITY_PNG = PNG_DIR / "equity_compare.png"
METRICS_PNG = PNG_DIR / "metrics_compare.png"
AVG_WEIGHTS_PNG = PNG_DIR / "average_active_weights_table.png"

EQUITY_CSV = CSV_DIR / "equity_curves.csv"
METRICS_CSV = CSV_DIR / "metrics_compare.csv"
AVG_WEIGHTS_CSV = CSV_DIR / "average_active_weights.csv"
REBALANCE_EVENTS_CSV = CSV_DIR / "rebalance_events.csv"

INVVOL30_WEIGHTS_CSV = CSV_DIR / "weights_tpi_sleeve_invvol30.csv"
INVVOL90_WEIGHTS_CSV = CSV_DIR / "weights_tpi_sleeve_invvol90.csv"
STANDARD_WEIGHTS_CSV = CSV_DIR / "weights_tpi_sleeve_standard.csv"

INVVOL30_TRADES_CSV = CSV_DIR / "trades_tpi_sleeve_invvol30.csv"
INVVOL90_TRADES_CSV = CSV_DIR / "trades_tpi_sleeve_invvol90.csv"
STANDARD_TRADES_CSV = CSV_DIR / "trades_tpi_sleeve_standard.csv"
BUYHOLD_TRADES_CSV = CSV_DIR / "trades_buy_hold_btc.csv"


# =========================================================
# BASIC HELPERS
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


# =========================================================
# LOADING
# =========================================================
def load_two_col_csv(path, out_col, value_candidates):
    path = require_file(path)
    df = pd.read_csv(path)

    cols = {c.lower().strip(): c for c in df.columns}

    date_col = None
    for c in ["time", "date", "datetime", "timestamp"]:
        if c in cols:
            date_col = cols[c]
            break
    if date_col is None:
        raise ValueError(f"{path}: missing date column")

    value_col = None
    for c in value_candidates:
        if c in cols:
            value_col = cols[c]
            break
    if value_col is None:
        raise ValueError(f"{path}: missing value column. Expected one of {value_candidates}")

    out = df[[date_col, value_col]].copy()
    out.columns = ["time", out_col]
    out["time"] = pd.to_datetime(out["time"], dayfirst=True, errors="coerce")
    out[out_col] = out[out_col].astype(str).str.replace(",", ".", regex=False)
    out[out_col] = pd.to_numeric(out[out_col], errors="coerce")
    out = out.dropna(subset=["time"]).drop_duplicates(subset=["time"]).sort_values("time")
    out = out.set_index("time")

    if START_DATE is not None:
        out = out.loc[out.index >= pd.Timestamp(START_DATE)]
    if END_DATE is not None:
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
            raise RuntimeError(f"Yahoo Finance data for {ticker} does not contain Close")
    else:
        if "Close" not in df.columns:
            raise RuntimeError(f"Yahoo Finance data for {ticker} does not contain Close")
        out = df[["Close"]].copy()

    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    return out


def load_all_tpis(base_dir):
    tpis = {}
    for name, fn in TPI_FILES.items():
        full = base_dir / fn
        tpis[name] = load_tpi_csv(full, name).iloc[:, 0]
    return tpis


def load_all_prices(base_dir):
    price_frames = {}

    for asset in ASSETS:
        if asset in LOCAL_PRICE_FILES:
            full = base_dir / LOCAL_PRICE_FILES[asset]
            s = load_price_csv(full, asset).iloc[:, 0]
            price_frames[asset] = s
        else:
            yf_df = safe_download_yf_close(YF_TICKERS[asset], start=START_DATE, end=END_DATE)
            s = yf_df["Close"].rename(asset)
            price_frames[asset] = s

    prices = pd.concat(price_frames.values(), axis=1).sort_index()

    if START_DATE is not None:
        prices = prices.loc[prices.index >= pd.Timestamp(START_DATE)]
    if END_DATE is not None:
        prices = prices.loc[prices.index <= pd.Timestamp(END_DATE)]

    if prices.empty:
        raise RuntimeError("Price dataframe is empty after loading data")

    return prices


def build_master_dataframe(tpis, prices):
    all_series = [prices] + [s.to_frame(name=s.name) for s in tpis.values()]
    df = pd.concat(all_series, axis=1).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.loc[df.index.notna()]

    if START_DATE is not None:
        df = df.loc[df.index >= pd.Timestamp(START_DATE)]
    if END_DATE is not None:
        df = df.loc[df.index <= pd.Timestamp(END_DATE)]

    if df.empty:
        raise RuntimeError("Combined master dataframe is empty")
    return df


# =========================================================
# TPI SLEEVE LOGIC
# =========================================================
def first_row_of_month(index):
    s = pd.Series(index.to_period("M"), index=index)
    return ~s.duplicated()


def build_eligibility_table(returns, total_tpi, asset_tpis):
    idx = returns.index
    assets = list(returns.columns)

    total_long = total_tpi.reindex(idx).ffill() >= TOTAL_THRESHOLD

    eligible = pd.DataFrame(False, index=idx, columns=assets)
    for a in assets:
        asset_sig = asset_tpis[a].reindex(idx).ffill() >= TOTAL_THRESHOLD
        eligible[a] = total_long & asset_sig

    live = returns.notna()
    return total_long, eligible, live


def rolling_inv_vol_scores(returns, window):
    scores = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
    for col in returns.columns:
        vol = returns[col].rolling(window).std(ddof=1)
        score = 1.0 / vol
        score = score.replace([np.inf, -np.inf], np.nan)
        scores[col] = score
    return scores


def normalize_positive_scores(score_row, eligible_row, live_row):
    score_row = score_row.copy()
    allowed = eligible_row.astype(bool) & live_row.astype(bool)
    score_row[~allowed] = np.nan

    positive = score_row.clip(lower=0.0)
    total = positive.sum(skipna=True)

    weights = pd.Series(0.0, index=score_row.index)
    if pd.notna(total) and total > 0:
        weights = positive.fillna(0.0) / total

    return weights


def standard_weights_with_cash(base_weights, eligible_row, live_row):
    allowed = eligible_row.astype(bool) & live_row.astype(bool)
    w = base_weights.reindex(allowed.index).fillna(0.0).copy()
    w[~allowed] = 0.0
    return w


def signals_materially_changed(current_eligible, previous_eligible):
    if previous_eligible is None:
        return True
    return not current_eligible.fillna(False).equals(previous_eligible.fillna(False))


def build_tpi_sleeve_invvol_weights(returns, total_tpi, asset_tpis, window):
    idx = returns.index
    assets = list(returns.columns)

    total_long, eligible, live = build_eligibility_table(returns, total_tpi, asset_tpis)
    scores = rolling_inv_vol_scores(returns.fillna(0.0), window=window)
    rebalance_month = first_row_of_month(idx)

    weights = pd.DataFrame(0.0, index=idx, columns=assets)
    prev_w = pd.Series(0.0, index=assets)
    prev_eligible = None

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
                curr_scores = scores.loc[dt]
                curr_w = normalize_positive_scores(curr_scores, eligible_today, live_today)
                if curr_w.sum() <= 0:
                    curr_w = pd.Series(0.0, index=assets)
            else:
                curr_w = prev_w.copy()
                invalid_now = ~(eligible_today & live_today)
                curr_w[invalid_now] = 0.0
                if curr_w.sum() > 0:
                    curr_w = curr_w / curr_w.sum()

        weights.loc[dt] = curr_w.values
        prev_w = curr_w
        prev_eligible = eligible_today.copy()

    return weights


def build_tpi_sleeve_standard_weights(returns, total_tpi, asset_tpis, base_weights):
    idx = returns.index
    assets = list(returns.columns)

    total_long, eligible, live = build_eligibility_table(returns, total_tpi, asset_tpis)
    rebalance_month = first_row_of_month(idx)

    weights = pd.DataFrame(0.0, index=idx, columns=assets)
    prev_w = pd.Series(0.0, index=assets)
    prev_eligible = None

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
                curr_w = standard_weights_with_cash(base_weights, eligible_today, live_today)
            else:
                curr_w = prev_w.copy()
                invalid_now = ~(eligible_today & live_today)
                curr_w[invalid_now] = 0.0

        weights.loc[dt] = curr_w.values
        prev_w = curr_w
        prev_eligible = eligible_today.copy()

    return weights


# =========================================================
# REBALANCE EVENTS
# =========================================================
def get_rebalance_dates(weights, tol=1e-12):
    w = weights.fillna(0.0).copy()
    changed = w.diff().abs().sum(axis=1) > tol
    if len(changed) > 0:
        changed.iloc[0] = w.iloc[0].abs().sum() > tol
    return weights.index[changed]


def build_rebalance_events_table(rebalance_dates_map):
    rows = []
    for portfolio, dates in rebalance_dates_map.items():
        for dt in pd.DatetimeIndex(dates):
            rows.append({
                "Portfolio": portfolio,
                "Rebalance Date": pd.Timestamp(dt)
            })
    out = pd.DataFrame(rows).sort_values(["Portfolio", "Rebalance Date"]).reset_index(drop=True)
    return out


# =========================================================
# EQUITY / TRADES
# =========================================================
def turnover_from_weights(weights):
    return weights.fillna(0.0).diff().abs().sum(axis=1).fillna(weights.iloc[0].abs().sum())


def compute_equity_from_weights(returns, weights):
    aligned_r = returns.reindex(weights.index)
    effective_w = weights.shift(1).fillna(0.0)

    gross_ret = (effective_w * aligned_r.fillna(0.0)).sum(axis=1)
    turnover = turnover_from_weights(weights)
    fees = turnover * FEE_PER_TURNOVER
    net_ret = gross_ret - fees

    equity = INITIAL_CAPITAL * (1.0 + net_ret.fillna(0.0)).cumprod()
    return net_ret, equity


def compute_buy_hold_equity_from_price(price_series):
    p = price_series.dropna().copy()
    if p.empty:
        raise RuntimeError("Benchmark price series is empty")

    equity = INITIAL_CAPITAL * (p / p.iloc[0])
    daily_ret = p.pct_change().fillna(0.0)
    return daily_ret, equity


def extract_trades_from_weights(equity, weights):
    exposure = weights.sum(axis=1).fillna(0.0)
    invested = exposure > 0

    entries = invested & (~invested.shift(1).fillna(False))
    exits = (~invested) & invested.shift(1).fillna(False)

    entry_dates = list(equity.index[entries])
    exit_dates = list(equity.index[exits])

    if len(entry_dates) > len(exit_dates):
        exit_dates.append(equity.index[-1])

    trades = []
    for ent, ex in zip(entry_dates, exit_dates):
        eq_slice = equity.loc[(equity.index >= ent) & (equity.index <= ex)]
        if len(eq_slice) < 2:
            continue

        entry_eq = float(eq_slice.iloc[0])
        exit_eq = float(eq_slice.iloc[-1])

        trades.append({
            "entry_date": ent,
            "exit_date": ex,
            "entry_equity": entry_eq,
            "exit_equity": exit_eq,
            "profit_dollar": exit_eq - entry_eq,
            "trade_return": exit_eq / entry_eq - 1.0,
            "days": (ex - ent).days,
        })

    return pd.DataFrame(trades)


def extract_single_buy_hold_trade(equity):
    eq = equity.dropna()
    if len(eq) < 2:
        return pd.DataFrame(columns=[
            "entry_date", "exit_date", "entry_equity", "exit_equity",
            "profit_dollar", "trade_return", "days"
        ])

    entry_eq = float(eq.iloc[0])
    exit_eq = float(eq.iloc[-1])

    return pd.DataFrame([{
        "entry_date": eq.index[0],
        "exit_date": eq.index[-1],
        "entry_equity": entry_eq,
        "exit_equity": exit_eq,
        "profit_dollar": exit_eq - entry_eq,
        "trade_return": exit_eq / entry_eq - 1.0,
        "days": (eq.index[-1] - eq.index[0]).days,
    }])


# =========================================================
# METRICS
# =========================================================
def max_drawdown_pct(equity):
    equity = equity.dropna()
    if equity.empty:
        return np.nan
    running_peak = equity.cummax()
    drawdown = (equity - running_peak) / running_peak * 100.0
    return drawdown.min()


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
    downside_sq_mean = np.mean(np.square(downside))
    downside_dev = math.sqrt(downside_sq_mean)
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
    elif gains > 0:
        return np.inf
    return np.nan


def profit_factor(trades):
    if trades.empty:
        return np.nan
    gross_profit = trades.loc[trades["profit_dollar"] > 0, "profit_dollar"].sum()
    gross_loss = -trades.loc[trades["profit_dollar"] < 0, "profit_dollar"].sum()

    if gross_loss > 0:
        return gross_profit / gross_loss
    elif gross_profit > 0:
        return np.inf
    return np.nan


def win_rate_pct(trades):
    if trades.empty:
        return np.nan
    return (trades["profit_dollar"] > 0).mean() * 100.0


def build_metrics(name, equity, daily_returns, trades):
    return {
        "Portfolio": name,
        "Max Equity Drawdown %": round2_or_nan(max_drawdown_pct(equity)),
        "Sharpe Ratio": round2_or_nan(sharpe_ratio(daily_returns)),
        "Sortino Ratio": round2_or_nan(sortino_ratio(daily_returns)),
        "Omega Ratio": round2_or_nan(omega_ratio(daily_returns, threshold=0.0)),
        "Profit Factor": round2_or_nan(profit_factor(trades)),
        "Win Rate %": round2_or_nan(win_rate_pct(trades)),
        "Number of Trades": int(len(trades)),
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
# OUTPUT / VISUALS
# =========================================================
def export_table_png(df, path, caption, percent_cols=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    percent_cols = set(percent_cols or [])
    export_df = df.copy()

    for col in export_df.columns:
        if pd.api.types.is_numeric_dtype(export_df[col]):
            if col in percent_cols:
                export_df[col] = export_df[col].apply(
                    lambda x: "" if pd.isna(x) else f"{x:.2%}"
                )
            else:
                export_df[col] = export_df[col].apply(
                    lambda x: "" if pd.isna(x) else f"{x:,.2f}"
                )

    nrows, ncols = export_df.shape
    fig_width = max(10, min(24, 1.55 * ncols + 2))
    fig_height = max(3.0, min(18, 0.50 * (nrows + 2) + 1.0))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.set_title(caption, fontsize=14, fontweight="bold", pad=14)

    table = ax.table(
        cellText=export_df.values,
        colLabels=export_df.columns,
        loc="center",
        cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.35)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#1f2937")
        else:
            cell.set_facecolor("#f8fafc" if row % 2 == 0 else "white")
        cell.set_edgecolor("#d1d5db")

    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_equity_curves(curves_df, rebalance_dates_map, path, title):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 7))

    colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#ea580c", "#0891b2"]
    markers = ["o", "s", "^", "D", "P", "X"]

    for i, col in enumerate(curves_df.columns):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        ax.plot(curves_df.index, curves_df[col], label=col, linewidth=2, color=color)

        reb_dates = rebalance_dates_map.get(col, pd.DatetimeIndex([]))
        reb_dates = pd.DatetimeIndex([d for d in reb_dates if d in curves_df.index])

        if len(reb_dates) > 0:
            yvals = curves_df.loc[reb_dates, col]
            ax.scatter(
                reb_dates,
                yvals,
                s=28,
                marker=marker,
                color=color,
                edgecolor="white",
                linewidth=0.6,
                alpha=0.95,
                zorder=5
            )

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


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
    returns = prices.pct_change()

    total_tpi = master["TOTAL_TPI"].copy()
    asset_tpis = {a: master[f"{a}_TPI"].copy() for a in ASSETS}

    portfolio_results = []

    # 1) TPI Sleeve Inv Vol 30D
    w_inv30 = build_tpi_sleeve_invvol_weights(
        returns=returns,
        total_tpi=total_tpi,
        asset_tpis=asset_tpis,
        window=30
    )
    r_inv30, e_inv30 = compute_equity_from_weights(returns, w_inv30)
    t_inv30 = extract_trades_from_weights(e_inv30, w_inv30)
    safe_to_csv(w_inv30, INVVOL30_WEIGHTS_CSV, index=True)
    safe_to_csv(t_inv30, INVVOL30_TRADES_CSV, index=False)
    portfolio_results.append({
        "name": "TPI Sleeve (Inv Vol 30D)",
        "weights": w_inv30,
        "daily_ret": r_inv30,
        "equity": e_inv30,
        "trades": t_inv30,
    })

    # 2) TPI Sleeve Inv Vol 90D
    w_inv90 = build_tpi_sleeve_invvol_weights(
        returns=returns,
        total_tpi=total_tpi,
        asset_tpis=asset_tpis,
        window=90
    )
    r_inv90, e_inv90 = compute_equity_from_weights(returns, w_inv90)
    t_inv90 = extract_trades_from_weights(e_inv90, w_inv90)
    safe_to_csv(w_inv90, INVVOL90_WEIGHTS_CSV, index=True)
    safe_to_csv(t_inv90, INVVOL90_TRADES_CSV, index=False)
    portfolio_results.append({
        "name": "TPI Sleeve (Inv Vol 90D)",
        "weights": w_inv90,
        "daily_ret": r_inv90,
        "equity": e_inv90,
        "trades": t_inv90,
    })

    # 3) TPI Sleeve Standard Fixed Weights with Cash Residual
    w_std = build_tpi_sleeve_standard_weights(
        returns=returns,
        total_tpi=total_tpi,
        asset_tpis=asset_tpis,
        base_weights=STANDARD_TARGET_WEIGHTS
    )
    r_std, e_std = compute_equity_from_weights(returns, w_std)
    t_std = extract_trades_from_weights(e_std, w_std)
    safe_to_csv(w_std, STANDARD_WEIGHTS_CSV, index=True)
    safe_to_csv(t_std, STANDARD_TRADES_CSV, index=False)
    portfolio_results.append({
        "name": "TPI Sleeve (Standard Weights + Cash)",
        "weights": w_std,
        "daily_ret": r_std,
        "equity": e_std,
        "trades": t_std,
    })

    # 4) Buy & Hold BTC
    bench_price = prices[BENCHMARK_ASSET].dropna().copy()
    bench_daily_ret, bench_equity = compute_buy_hold_equity_from_price(bench_price)
    bench_weights = pd.DataFrame(0.0, index=bench_equity.index, columns=ASSETS)
    bench_weights[BENCHMARK_ASSET] = 1.0
    bench_trades = extract_single_buy_hold_trade(bench_equity)
    safe_to_csv(bench_trades, BUYHOLD_TRADES_CSV, index=False)
    portfolio_results.append({
        "name": f"Buy & Hold {BENCHMARK_ASSET}",
        "weights": bench_weights,
        "daily_ret": bench_daily_ret,
        "equity": bench_equity,
        "trades": bench_trades,
    })

    rebalance_dates_map = {
        "TPI Sleeve (Inv Vol 30D)": get_rebalance_dates(w_inv30),
        "TPI Sleeve (Inv Vol 90D)": get_rebalance_dates(w_inv90),
        "TPI Sleeve (Standard Weights + Cash)": get_rebalance_dates(w_std),
        f"Buy & Hold {BENCHMARK_ASSET}": get_rebalance_dates(bench_weights),
    }

    rebalance_events_df = build_rebalance_events_table(rebalance_dates_map)
    safe_to_csv(rebalance_events_df, REBALANCE_EVENTS_CSV, index=False)

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
    plot_equity_curves(
        equity_df,
        rebalance_dates_map,
        EQUITY_PNG,
        "Portfolio Equity Comparison with Rebalance Events"
    )

    avg_active_weights_df = pd.DataFrame(
        [average_active_weights(p["weights"]).rename(p["name"]) for p in portfolio_results]
    ).fillna(0.0)
    avg_active_weights_df.index.name = "Portfolio"
    safe_to_csv(avg_active_weights_df, AVG_WEIGHTS_CSV, index=True)

    export_table_png(
        avg_active_weights_df.reset_index(),
        AVG_WEIGHTS_PNG,
        "Average Active Weights",
        percent_cols=set(ASSETS)
    )

    print("\nMetrics:")
    print(metrics_df.to_string(index=False))

    print("\nAverage Active Weights:")
    print(avg_active_weights_df.to_string())

    print("\nRebalance event counts:")
    for name, dates in rebalance_dates_map.items():
        print(f"{name}: {len(dates)}")

    print("\nSaved files:")
    for p in [
        EQUITY_PNG,
        METRICS_PNG,
        AVG_WEIGHTS_PNG,
        EQUITY_CSV,
        METRICS_CSV,
        AVG_WEIGHTS_CSV,
        REBALANCE_EVENTS_CSV,
        INVVOL30_WEIGHTS_CSV,
        INVVOL90_WEIGHTS_CSV,
        STANDARD_WEIGHTS_CSV,
        INVVOL30_TRADES_CSV,
        INVVOL90_TRADES_CSV,
        STANDARD_TRADES_CSV,
        BUYHOLD_TRADES_CSV,
    ]:
        print(p)


if __name__ == "__main__":
    main()
