"""
Plan 10: Options & Implied Volatility Analysis

Analyzes ONDS options data to extract IV-derived features
and tests their predictive power.
"""
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import TARGET_TICKER, DATA_RAW, DATA_PROC, FIGURES_DIR


def load_options_data(ticker: str = TARGET_TICKER) -> pd.DataFrame:
    """Load saved options data."""
    path = DATA_RAW / f"options_all_{ticker}.csv"
    if not path.exists():
        print(f"  No options data for {ticker}. Run collectors/options.py first.")
        return pd.DataFrame()
    return pd.read_csv(path)


def compute_iv_summary(options_df: pd.DataFrame, current_price: float) -> dict:
    """
    Compute summary IV statistics from options chain.
    """
    if options_df.empty:
        return {}

    summary = {}

    # Split calls and puts
    calls = options_df[options_df["type"] == "call"] if "type" in options_df.columns else pd.DataFrame()
    puts = options_df[options_df["type"] == "put"] if "type" in options_df.columns else pd.DataFrame()

    # Overall IV statistics
    if "impliedVolatility" in options_df.columns:
        iv = options_df["impliedVolatility"].dropna()
        summary["mean_IV"] = iv.mean()
        summary["median_IV"] = iv.median()
        summary["min_IV"] = iv.min()
        summary["max_IV"] = iv.max()

    # ATM IV (nearest strike to current price)
    if "strike" in options_df.columns and "impliedVolatility" in options_df.columns:
        for label, df in [("call", calls), ("put", puts)]:
            if df.empty:
                continue
            near = (df["strike"] - current_price).abs()
            atm_idx = near.idxmin()
            summary[f"ATM_IV_{label}"] = df.loc[atm_idx, "impliedVolatility"]
            summary[f"ATM_strike_{label}"] = df.loc[atm_idx, "strike"]

    # Put-Call ratios
    if not calls.empty and not puts.empty:
        if "volume" in calls.columns:
            cv = calls["volume"].sum()
            pv = puts["volume"].sum()
            summary["PC_volume_ratio"] = pv / cv if cv > 0 else np.nan

        if "openInterest" in calls.columns:
            coi = calls["openInterest"].sum()
            poi = puts["openInterest"].sum()
            summary["PC_OI_ratio"] = poi / coi if coi > 0 else np.nan

    # IV Skew: OTM put IV vs ATM call IV
    if not puts.empty and "impliedVolatility" in puts.columns:
        otm_puts = puts[puts["strike"] < current_price * 0.95]
        if not otm_puts.empty:
            summary["OTM_put_IV_avg"] = otm_puts["impliedVolatility"].mean()
            if "ATM_IV_call" in summary:
                summary["IV_skew"] = summary["OTM_put_IV_avg"] - summary["ATM_IV_call"]

    return summary


def plot_iv_smile(options_df: pd.DataFrame, current_price: float, save: bool = True):
    """Plot IV smile for each expiration."""
    if options_df.empty or "impliedVolatility" not in options_df.columns:
        print("  No IV data to plot.")
        return

    expirations = sorted(options_df["expiration"].unique())[:4]  # First 4 expirations

    fig, axes = plt.subplots(1, len(expirations), figsize=(5 * len(expirations), 5))
    if len(expirations) == 1:
        axes = [axes]

    for ax, exp in zip(axes, expirations):
        exp_data = options_df[options_df["expiration"] == exp]

        calls = exp_data[exp_data["type"] == "call"].sort_values("strike")
        puts = exp_data[exp_data["type"] == "put"].sort_values("strike")

        if not calls.empty:
            ax.plot(calls["strike"], calls["impliedVolatility"], "b.-", label="Calls", alpha=0.7)
        if not puts.empty:
            ax.plot(puts["strike"], puts["impliedVolatility"], "r.-", label="Puts", alpha=0.7)

        ax.axvline(current_price, color="gray", linestyle="--", alpha=0.5, label="Spot")
        ax.set_title(f"Exp: {exp}")
        ax.set_xlabel("Strike")
        ax.set_ylabel("Implied Volatility")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"ONDS IV Smile (Current Price: ${current_price:.2f})", fontsize=12)
    plt.tight_layout()

    if save:
        fig.savefig(FIGURES_DIR / "onds_iv_smile.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: {FIGURES_DIR / 'onds_iv_smile.png'}")
    plt.close(fig)


def plot_iv_surface(options_df: pd.DataFrame, current_price: float, save: bool = True):
    """Plot 3D IV surface (strike x expiration x IV)."""
    if options_df.empty or "impliedVolatility" not in options_df.columns:
        return

    calls = options_df[options_df["type"] == "call"].copy()
    if calls.empty:
        return

    calls["moneyness"] = calls["strike"] / current_price
    calls["expiration_dt"] = pd.to_datetime(calls["expiration"])
    calls["dte"] = (calls["expiration_dt"] - pd.Timestamp.now()).dt.days

    # Filter reasonable range
    calls = calls[(calls["moneyness"] > 0.7) & (calls["moneyness"] < 1.3)]
    calls = calls[calls["impliedVolatility"] > 0]

    if len(calls) < 10:
        print("  Not enough data for IV surface plot.")
        return

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        calls["moneyness"],
        calls["dte"],
        calls["impliedVolatility"],
        c=calls["impliedVolatility"],
        cmap="viridis",
        alpha=0.6,
    )
    ax.set_xlabel("Moneyness (K/S)")
    ax.set_ylabel("Days to Expiry")
    ax.set_zlabel("Implied Volatility")
    ax.set_title("ONDS Implied Volatility Surface")
    fig.colorbar(sc, ax=ax, shrink=0.5, label="IV")

    if save:
        fig.savefig(FIGURES_DIR / "onds_iv_surface.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: {FIGURES_DIR / 'onds_iv_surface.png'}")
    plt.close(fig)
