"""
Plan 1: Technical Analysis for ONDS

Computes standard technical indicators and tests their predictive power
for future returns. Generates signals for backtesting.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import TA_PARAMS, DATA_PROC, FIGURES_DIR
import matplotlib.pyplot as plt


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicators on OHLCV data.
    Returns DataFrame with all indicators added as columns.
    """
    out = df.copy()
    p = TA_PARAMS

    close = out["Close"]
    high  = out["High"]
    low   = out["Low"]
    volume = out["Volume"]

    # ── RSI ────────────────────────────────────────────────────
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=p["rsi_period"], min_periods=p["rsi_period"]).mean()
    avg_loss = loss.ewm(span=p["rsi_period"], min_periods=p["rsi_period"]).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["RSI"] = 100 - (100 / (1 + rs))

    # ── MACD ───────────────────────────────────────────────────
    ema_fast = close.ewm(span=p["macd_fast"]).mean()
    ema_slow = close.ewm(span=p["macd_slow"]).mean()
    out["MACD"]        = ema_fast - ema_slow
    out["MACD_signal"] = out["MACD"].ewm(span=p["macd_signal"]).mean()
    out["MACD_hist"]   = out["MACD"] - out["MACD_signal"]

    # ── Bollinger Bands ────────────────────────────────────────
    sma = close.rolling(p["bb_period"]).mean()
    std = close.rolling(p["bb_period"]).std()
    out["BB_upper"] = sma + p["bb_std"] * std
    out["BB_lower"] = sma - p["bb_std"] * std
    out["BB_pct"]   = (close - out["BB_lower"]) / (out["BB_upper"] - out["BB_lower"])

    # ── ATR (Average True Range) ───────────────────────────────
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    out["ATR"] = tr.rolling(p["atr_period"]).mean()
    out["ATR_pct"] = out["ATR"] / close  # normalized

    # ── OBV (On-Balance Volume) ────────────────────────────────
    obv_sign = np.sign(close.diff()).fillna(0)
    out["OBV"] = (obv_sign * volume).cumsum()
    out["OBV_MA"] = out["OBV"].rolling(p["obv_ma"]).mean()

    # ── SMA crossover ──────────────────────────────────────────
    out["SMA_short"] = close.rolling(p["sma_short"]).mean()
    out["SMA_long"]  = close.rolling(p["sma_long"]).mean()
    out["SMA_cross"]  = (out["SMA_short"] > out["SMA_long"]).astype(int)

    # ── Volume ratio ───────────────────────────────────────────
    out["Vol_ratio"] = volume / volume.rolling(20).mean()

    # ── Returns ────────────────────────────────────────────────
    out["Return_1d"]  = close.pct_change(1)
    out["Return_5d"]  = close.pct_change(5)
    out["Return_20d"] = close.pct_change(20)
    out["Fwd_1d"]     = close.pct_change(1).shift(-1)  # forward return
    out["Fwd_5d"]     = close.pct_change(5).shift(-5)

    return out


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate trading signals from technical indicators.
    Combines multiple indicators into a composite score.
    """
    sig = pd.DataFrame(index=df.index)

    # RSI signal: oversold < 30 → buy, overbought > 70 → sell
    sig["rsi_sig"] = 0
    sig.loc[df["RSI"] < 30, "rsi_sig"] = 1
    sig.loc[df["RSI"] > 70, "rsi_sig"] = -1

    # MACD signal: histogram crossing zero
    sig["macd_sig"] = np.sign(df["MACD_hist"])

    # Bollinger signal: below lower → buy, above upper → sell
    sig["bb_sig"] = 0
    sig.loc[df["BB_pct"] < 0, "bb_sig"] = 1
    sig.loc[df["BB_pct"] > 1, "bb_sig"] = -1

    # SMA crossover
    sig["sma_sig"] = df["SMA_cross"] * 2 - 1  # 1 or -1

    # Volume surge + price direction
    sig["vol_sig"] = 0
    vol_surge = df["Vol_ratio"] > 2
    sig.loc[vol_surge & (df["Return_1d"] > 0), "vol_sig"] = 1
    sig.loc[vol_surge & (df["Return_1d"] < 0), "vol_sig"] = -1

    # Composite: equal-weight average, then discretize
    sig["composite"] = sig[["rsi_sig", "macd_sig", "bb_sig", "sma_sig", "vol_sig"]].mean(axis=1)
    sig["signal"] = np.sign(sig["composite"])

    return sig


def analyze_predictive_power(df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """
    Test which indicators have predictive power for next-day returns.
    Uses rank correlation (Spearman) between indicator values and forward returns.
    """
    from scipy import stats

    indicators = ["RSI", "MACD_hist", "BB_pct", "ATR_pct", "Vol_ratio",
                  "SMA_cross", "OBV"]
    target = "Fwd_1d"

    results = []
    for ind in indicators:
        valid = df[[ind, target]].dropna()
        if len(valid) < 30:
            continue
        corr, pval = stats.spearmanr(valid[ind], valid[target])
        results.append({
            "Indicator": ind,
            "Spearman_corr": round(corr, 4),
            "p_value": round(pval, 4),
            "significant": "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else "",
            "n_obs": len(valid),
        })

    result_df = pd.DataFrame(results).sort_values("p_value")

    if save:
        result_df.to_csv(DATA_PROC / "technical_predictive_power.csv", index=False)

    print("\n  Technical Indicator Predictive Power (Spearman vs Fwd_1d):")
    print("  " + "-" * 60)
    for _, row in result_df.iterrows():
        print(f"    {row['Indicator']:15s}  corr={row['Spearman_corr']:+.4f}  "
              f"p={row['p_value']:.4f} {row['significant']}")

    return result_df


def plot_technical_dashboard(df: pd.DataFrame, save: bool = True):
    """Plot ONDS price with key technical overlays."""
    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True,
                              height_ratios=[3, 1, 1, 1])

    # Price + Bollinger Bands + SMA
    ax = axes[0]
    ax.plot(df.index, df["Close"], label="Close", color="#1976D2", linewidth=1)
    ax.plot(df.index, df["BB_upper"], color="#E0E0E0", linewidth=0.5)
    ax.plot(df.index, df["BB_lower"], color="#E0E0E0", linewidth=0.5)
    ax.fill_between(df.index, df["BB_upper"], df["BB_lower"], alpha=0.1, color="#90CAF9")
    ax.plot(df.index, df["SMA_short"], label=f"SMA{TA_PARAMS['sma_short']}",
            color="#FF9800", linewidth=0.8, alpha=0.7)
    ax.plot(df.index, df["SMA_long"], label=f"SMA{TA_PARAMS['sma_long']}",
            color="#F44336", linewidth=0.8, alpha=0.7)
    ax.set_ylabel("Price ($)")
    ax.set_title("ONDS Technical Analysis Dashboard")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    # RSI
    ax = axes[1]
    ax.plot(df.index, df["RSI"], color="#9C27B0", linewidth=0.8)
    ax.axhline(70, color="red", linestyle="--", alpha=0.5)
    ax.axhline(30, color="green", linestyle="--", alpha=0.5)
    ax.fill_between(df.index, 30, 70, alpha=0.05, color="gray")
    ax.set_ylabel("RSI")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    # MACD
    ax = axes[2]
    ax.plot(df.index, df["MACD"], label="MACD", color="#2196F3", linewidth=0.8)
    ax.plot(df.index, df["MACD_signal"], label="Signal", color="#FF5722", linewidth=0.8)
    colors = ["#4CAF50" if v >= 0 else "#F44336" for v in df["MACD_hist"]]
    ax.bar(df.index, df["MACD_hist"], color=colors, alpha=0.5, width=1)
    ax.set_ylabel("MACD")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Volume
    ax = axes[3]
    colors = ["#4CAF50" if r >= 0 else "#F44336" for r in df["Return_1d"]]
    ax.bar(df.index, df["Volume"], color=colors, alpha=0.5, width=1)
    ax.set_ylabel("Volume")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "onds_technical_dashboard.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: {FIGURES_DIR / 'onds_technical_dashboard.png'}")
    plt.close(fig)
