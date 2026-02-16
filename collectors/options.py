"""
Plan 10: Options Data Collection for ONDS

Collects options chains from yfinance and computes IV-derived features.
"""
import yfinance as yf
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import TARGET_TICKER, DATA_RAW


def collect_options_chain(ticker: str = TARGET_TICKER, save: bool = True) -> dict:
    """
    Collect all available options chains for a ticker.
    Returns dict with expiry dates as keys, (calls_df, puts_df) as values.
    """
    print(f"  Collecting options chain for {ticker}...")

    stock = yf.Ticker(ticker)
    expirations = stock.options

    if not expirations:
        print(f"    No options available for {ticker}")
        return {}

    print(f"    Found {len(expirations)} expiration dates: {expirations[:5]}...")

    all_calls = []
    all_puts = []

    for exp in expirations:
        try:
            chain = stock.option_chain(exp)
            calls = chain.calls.copy()
            puts = chain.puts.copy()
            calls["expiration"] = exp
            puts["expiration"] = exp
            calls["type"] = "call"
            puts["type"] = "put"
            all_calls.append(calls)
            all_puts.append(puts)
        except Exception as e:
            print(f"    Failed for {exp}: {e}")
            continue

    calls_df = pd.concat(all_calls, ignore_index=True) if all_calls else pd.DataFrame()
    puts_df = pd.concat(all_puts, ignore_index=True) if all_puts else pd.DataFrame()

    print(f"    Calls: {len(calls_df)} contracts")
    print(f"    Puts:  {len(puts_df)} contracts")

    if save:
        if not calls_df.empty:
            calls_df.to_csv(DATA_RAW / f"options_calls_{ticker}.csv", index=False)
        if not puts_df.empty:
            puts_df.to_csv(DATA_RAW / f"options_puts_{ticker}.csv", index=False)

        # Combined
        combined = pd.concat([calls_df, puts_df], ignore_index=True)
        combined.to_csv(DATA_RAW / f"options_all_{ticker}.csv", index=False)
        print(f"    Saved: {DATA_RAW / f'options_all_{ticker}.csv'}")

    return {"calls": calls_df, "puts": puts_df, "expirations": list(expirations)}


def compute_iv_features(calls_df: pd.DataFrame, puts_df: pd.DataFrame,
                        current_price: float) -> dict:
    """
    Compute IV-derived features from options chain snapshot.

    Features:
    - ATM IV (call and put, nearest strike to current price)
    - Put-call IV spread (ATM put IV - ATM call IV = skew proxy)
    - Put-call volume ratio
    - Put-call OI ratio
    - IV term structure slope (front-month vs back-month ATM IV)
    - 25-delta skew (if enough strikes)
    """
    features = {}

    if calls_df.empty or puts_df.empty:
        return features

    # ATM strike = nearest to current price
    for label, df in [("call", calls_df), ("put", puts_df)]:
        if "strike" not in df.columns or "impliedVolatility" not in df.columns:
            continue

        # Get nearest ATM for each expiration
        for exp in df["expiration"].unique():
            exp_df = df[df["expiration"] == exp]
            if exp_df.empty:
                continue
            atm_idx = (exp_df["strike"] - current_price).abs().idxmin()
            atm_row = exp_df.loc[atm_idx]
            features[f"{label}_ATM_IV_{exp}"] = atm_row.get("impliedVolatility", np.nan)

    # Put-call volume ratio (aggregate)
    total_call_vol = calls_df["volume"].sum() if "volume" in calls_df.columns else 0
    total_put_vol = puts_df["volume"].sum() if "volume" in puts_df.columns else 0
    features["put_call_volume_ratio"] = (
        total_put_vol / total_call_vol if total_call_vol > 0 else np.nan
    )

    # Put-call OI ratio
    total_call_oi = calls_df["openInterest"].sum() if "openInterest" in calls_df.columns else 0
    total_put_oi = puts_df["openInterest"].sum() if "openInterest" in puts_df.columns else 0
    features["put_call_oi_ratio"] = (
        total_put_oi / total_call_oi if total_call_oi > 0 else np.nan
    )

    # ATM IV spread (put - call) = skew proxy
    exps = sorted(calls_df["expiration"].unique())
    if exps:
        front_exp = exps[0]
        for label, df in [("call", calls_df), ("put", puts_df)]:
            front = df[df["expiration"] == front_exp]
            if not front.empty and "strike" in front.columns:
                atm_idx = (front["strike"] - current_price).abs().idxmin()
                features[f"{label}_ATM_IV_front"] = front.loc[atm_idx].get("impliedVolatility", np.nan)

        if "call_ATM_IV_front" in features and "put_ATM_IV_front" in features:
            features["IV_skew_front"] = features["put_ATM_IV_front"] - features["call_ATM_IV_front"]

        # Term structure: front vs back
        if len(exps) >= 2:
            back_exp = exps[-1]
            for label, df in [("call", calls_df), ("put", puts_df)]:
                back = df[df["expiration"] == back_exp]
                if not back.empty and "strike" in back.columns:
                    atm_idx = (back["strike"] - current_price).abs().idxmin()
                    features[f"{label}_ATM_IV_back"] = back.loc[atm_idx].get("impliedVolatility", np.nan)

            if "call_ATM_IV_front" in features and "call_ATM_IV_back" in features:
                features["IV_term_structure"] = (
                    features["call_ATM_IV_back"] - features["call_ATM_IV_front"]
                )

    return features


if __name__ == "__main__":
    print("=" * 60)
    print("OPTIONS DATA COLLECTION")
    print("=" * 60)
    data = collect_options_chain()
    if data:
        stock = yf.Ticker(TARGET_TICKER)
        hist = stock.history(period="1d")
        if not hist.empty:
            current_price = hist["Close"].iloc[-1]
            features = compute_iv_features(data["calls"], data["puts"], current_price)
            print(f"\n  IV Features (snapshot):")
            for k, v in features.items():
                if isinstance(v, float):
                    print(f"    {k}: {v:.4f}")
                else:
                    print(f"    {k}: {v}")
