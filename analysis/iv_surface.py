"""
IV Surface Analysis for ONDS
============================
Phase 2 of the major upgrade:

A. SSVI Model -- Gatheral & Jacquier (2014) arbitrage-free parametric
   volatility surface fitted to ONDS options (529 contracts, 10 expirations)

B. Daily IV-Proxy Features -- 7 VIX/RV-based features for backtesting
   (historical daily option chains unavailable, so we use market-wide proxies)

C. IV-Based Trading Strategies -- VRP mean-reversion, IV-regime conditional,
   vol surface composite

D. Integration with ML Pipeline -- test whether IV features improve
   walk-forward ML model performance
"""
import numpy as np
import pandas as pd
from scipy import optimize, stats
from pathlib import Path
from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.dates as mdates
import warnings
import sys

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    TARGET_TICKER, DATA_RAW, DATA_PROC, FIGURES_DIR, REPORTS_DIR, BACKTEST
)
from backtests.engine import backtest, print_results, compare_strategies
from analysis.advanced_research import (
    load_all_data, build_advanced_features, vol_adjusted_signal,
)
from analysis.robustness import bootstrap_sharpe, permutation_test
from analysis.param_optimization import make_split, period_backtest


# =====================================================================
# SECTION A: SSVI MODEL (Gatheral & Jacquier 2014)
# =====================================================================

def ssvi_total_variance(k: np.ndarray, theta: float, rho: float,
                         phi: float) -> np.ndarray:
    """
    SSVI formula for total implied variance w(k).

    w(k) = (theta/2) * (1 + rho*phi*k + sqrt((phi*k + rho)^2 + 1 - rho^2))

    Parameters
    ----------
    k : array of log-moneyness = ln(K/S)
    theta : ATM total variance (annualized IV^2 * tau)
    rho : skew parameter in [-1, 1]
    phi : vol-of-vol scaling

    Returns
    -------
    w : total variance at each moneyness point
    """
    inner = (phi * k + rho) ** 2 + (1 - rho ** 2)
    inner = np.maximum(inner, 1e-12)  # numerical safety
    w = (theta / 2.0) * (1.0 + rho * phi * k + np.sqrt(inner))
    return w


def ssvi_phi_power_law(tau: float, eta: float, gamma: float) -> float:
    """
    Power-law parameterization of phi (vol-of-vol).

    phi(tau) = eta / (tau^gamma * (1 + tau)^(1 - gamma))

    This ensures the surface is well-behaved across maturities.
    """
    return eta / (tau ** gamma * (1 + tau) ** (1 - gamma))


def ssvi_objective(params, k_array, tau_array, market_tv, theta_by_tau):
    """
    Least-squares objective for fitting SSVI parameters.

    Fits global parameters (rho, eta, gamma) while theta is per-expiration.
    """
    rho, eta, gamma = params

    # Constraints: |rho| < 1, eta > 0, 0 < gamma < 1
    if abs(rho) >= 0.999 or eta <= 0 or gamma <= 0 or gamma >= 1:
        return 1e10

    residuals = []
    for tau_val in np.unique(tau_array):
        mask = tau_array == tau_val
        k_slice = k_array[mask]
        tv_slice = market_tv[mask]
        theta = theta_by_tau.get(tau_val, tv_slice.mean())
        phi = ssvi_phi_power_law(tau_val, eta, gamma)

        model_tv = ssvi_total_variance(k_slice, theta, rho, phi)
        residuals.extend((model_tv - tv_slice).tolist())

    return np.sum(np.array(residuals) ** 2)


def fit_ssvi_surface(options_df: pd.DataFrame,
                     current_price: float) -> dict:
    """
    Fit SSVI surface to ONDS options data.

    Steps:
    1. Compute log-moneyness and total variance for each contract
    2. Estimate theta (ATM total variance) per expiration
    3. Fit global (rho, eta, gamma) via scipy.optimize.minimize
    4. Return fitted parameters and quality metrics

    Returns
    -------
    dict with keys:
        params: (rho, eta, gamma)
        theta_by_tau: dict mapping tau -> theta
        rmse: fit RMSE
        r2: R-squared
        residuals: array of fit residuals
    """
    if options_df.empty:
        return {"error": "No options data"}

    # Filter to contracts with valid IV
    df = options_df.copy()
    if "impliedVolatility" not in df.columns or "strike" not in df.columns:
        return {"error": "Missing impliedVolatility or strike columns"}

    df = df.dropna(subset=["impliedVolatility", "strike"])
    df = df[df["impliedVolatility"] > 0.01]  # remove near-zero IV
    df = df[df["impliedVolatility"] < 5.0]   # remove extreme IV

    if len(df) < 10:
        return {"error": f"Only {len(df)} valid contracts"}

    # Compute log-moneyness
    df["k"] = np.log(df["strike"] / current_price)

    # Compute time to expiration (in years)
    if "expiration" in df.columns:
        df["expiration_dt"] = pd.to_datetime(df["expiration"])
        # Approximate: use today as reference
        today = pd.Timestamp.now().normalize()
        df["tau"] = (df["expiration_dt"] - today).dt.days / 365.25
        df = df[df["tau"] > 0.01]  # Remove expired / near-expired
    elif "daysToExpiry" in df.columns:
        df["tau"] = df["daysToExpiry"] / 365.25
        df = df[df["tau"] > 0.01]
    else:
        return {"error": "No expiration date column found"}

    # Total variance = IV^2 * tau
    df["total_variance"] = df["impliedVolatility"] ** 2 * df["tau"]

    # Estimate theta (ATM total variance) per expiration
    theta_by_tau = {}
    for tau_val in df["tau"].unique():
        slice_df = df[df["tau"] == tau_val]
        # ATM = closest to k=0
        atm_idx = slice_df["k"].abs().idxmin()
        theta_by_tau[tau_val] = float(slice_df.loc[atm_idx, "total_variance"])

    k_array = df["k"].values
    tau_array = df["tau"].values
    market_tv = df["total_variance"].values

    # Fit SSVI parameters
    # Initial guess: rho=-0.3 (typical skew), eta=1.0, gamma=0.5
    result = optimize.minimize(
        ssvi_objective,
        x0=[-0.3, 1.0, 0.5],
        args=(k_array, tau_array, market_tv, theta_by_tau),
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-8},
    )

    rho, eta, gamma = result.x

    # Compute fitted values and residuals
    fitted_tv = []
    for i in range(len(k_array)):
        tau_val = tau_array[i]
        theta = theta_by_tau.get(tau_val, market_tv[i])
        phi = ssvi_phi_power_law(tau_val, eta, gamma)
        fitted_tv.append(ssvi_total_variance(
            np.array([k_array[i]]), theta, rho, phi
        )[0])
    fitted_tv = np.array(fitted_tv)
    residuals = market_tv - fitted_tv

    rmse = np.sqrt(np.mean(residuals ** 2))
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((market_tv - market_tv.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Compute ATM IV for RMSE context
    atm_iv = np.sqrt(np.mean(list(theta_by_tau.values())))
    rmse_pct = rmse / (atm_iv + 1e-6) * 100

    return {
        "params": {"rho": rho, "eta": eta, "gamma": gamma},
        "theta_by_tau": theta_by_tau,
        "rmse": rmse,
        "r2": r2,
        "rmse_pct_of_atm": rmse_pct,
        "n_contracts": len(df),
        "n_expirations": len(theta_by_tau),
        "residuals": residuals,
        "data": df,
        "fitted_tv": fitted_tv,
    }


def plot_ssvi_surface(fit_result: dict) -> None:
    """Plot fitted SSVI surface vs market data."""
    if "error" in fit_result:
        print(f"  Cannot plot: {fit_result['error']}")
        return

    df = fit_result["data"]
    params = fit_result["params"]
    theta_by_tau = fit_result["theta_by_tau"]

    fig = plt.figure(figsize=(16, 12))

    # --- 3D Surface ---
    ax1 = fig.add_subplot(221, projection="3d")

    # Market data points
    ax1.scatter(df["k"].values, df["tau"].values,
                df["total_variance"].values,
                c="red", s=15, alpha=0.6, label="Market")

    # Fitted surface grid
    k_grid = np.linspace(df["k"].min(), df["k"].max(), 50)
    tau_grid = np.linspace(df["tau"].min(), df["tau"].max(), 30)
    K, T = np.meshgrid(k_grid, tau_grid)
    W = np.zeros_like(K)

    for i, tau_val in enumerate(tau_grid):
        # Interpolate theta for this tau
        taus = sorted(theta_by_tau.keys())
        thetas = [theta_by_tau[t] for t in taus]
        theta_interp = np.interp(tau_val, taus, thetas)
        phi = ssvi_phi_power_law(tau_val, params["eta"], params["gamma"])
        W[i, :] = ssvi_total_variance(k_grid, theta_interp,
                                        params["rho"], phi)

    ax1.plot_surface(K, T, W, alpha=0.3, cmap="viridis")
    ax1.set_xlabel("Log-Moneyness (k)")
    ax1.set_ylabel("Time to Expiry (years)")
    ax1.set_zlabel("Total Variance")
    ax1.set_title("SSVI Fitted Surface")
    ax1.legend(fontsize=8)

    # --- Residuals ---
    ax2 = fig.add_subplot(222)
    ax2.scatter(df["k"].values, fit_result["residuals"],
                c=df["tau"].values, cmap="viridis", s=20, alpha=0.7)
    ax2.axhline(y=0, color="red", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Log-Moneyness (k)")
    ax2.set_ylabel("Residual (Market - Model)")
    ax2.set_title(f"Fit Residuals (RMSE={fit_result['rmse']:.4f}, "
                   f"R2={fit_result['r2']:.3f})")
    ax2.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax2.collections[0], ax=ax2)
    cbar.set_label("Tau (years)")

    # --- IV Smile per expiration ---
    ax3 = fig.add_subplot(223)
    taus = sorted(df["tau"].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(taus)))

    for tau_val, color in zip(taus, colors):
        mask = df["tau"] == tau_val
        k_slice = df.loc[mask, "k"].values
        iv_market = np.sqrt(df.loc[mask, "total_variance"].values / tau_val)
        sort_idx = np.argsort(k_slice)

        ax3.scatter(k_slice[sort_idx], iv_market[sort_idx],
                    c=[color], s=20, alpha=0.7)

        # Fitted smile
        k_fine = np.linspace(k_slice.min(), k_slice.max(), 50)
        theta = theta_by_tau[tau_val]
        phi = ssvi_phi_power_law(tau_val, params["eta"], params["gamma"])
        tv_fit = ssvi_total_variance(k_fine, theta, params["rho"], phi)
        iv_fit = np.sqrt(np.maximum(tv_fit / tau_val, 0))
        ax3.plot(k_fine, iv_fit, color=color,
                 label=f"tau={tau_val:.2f}y", linewidth=1.2)

    ax3.set_xlabel("Log-Moneyness (k)")
    ax3.set_ylabel("Implied Volatility")
    ax3.set_title("IV Smile by Expiration (Market dots, SSVI lines)")
    ax3.legend(fontsize=7, loc="upper right", ncol=2)
    ax3.grid(True, alpha=0.3)

    # --- Parameters summary ---
    ax4 = fig.add_subplot(224)
    ax4.axis("off")
    summary_text = (
        f"SSVI Fit Summary\n"
        f"{'='*35}\n"
        f"rho (skew):     {params['rho']:+.4f}\n"
        f"eta (vol-vol):  {params['eta']:.4f}\n"
        f"gamma (power):  {params['gamma']:.4f}\n"
        f"{'='*35}\n"
        f"RMSE:           {fit_result['rmse']:.6f}\n"
        f"R2:             {fit_result['r2']:.4f}\n"
        f"RMSE/ATM:       {fit_result['rmse_pct_of_atm']:.1f}%\n"
        f"Contracts:      {fit_result['n_contracts']}\n"
        f"Expirations:    {fit_result['n_expirations']}\n"
    )
    ax4.text(0.1, 0.5, summary_text, family="monospace",
             fontsize=11, verticalalignment="center",
             bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.3))

    plt.tight_layout()
    path = FIGURES_DIR / "ssvi_onds_surface.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# =====================================================================
# SECTION B: DAILY IV-PROXY FEATURES
# =====================================================================

def build_iv_proxy_features(data: dict) -> pd.DataFrame:
    """
    Build 7 daily IV-proxy features using VIX and realized volatility.

    Since historical daily ONDS option chains are unavailable, we use
    market-wide volatility proxies (VIX, RV) as IV stand-ins.
    All features are causal (use only past data at each point).

    Features:
    1. VRP:           VIX^2/252 - RV_20d^2   (variance risk premium)
    2. IV Rank (60d): percentile(VIX, 60d)    (where is IV vs recent)
    3. VIX Term:      VIX - VIX_MA5           (rising/falling vol)
    4. Skew Proxy:    rolling asymmetry        (tail risk)
    5. VIX Regime:    VIX > 80th pctl = 1      (high-vol flag)
    6. RV Ratio:      RV_5d / RV_20d           (vol acceleration)
    7. VIX-ONDS Corr: rolling 20d correlation  (flight-to-quality)
    """
    onds = data["onds"]
    closes = data["closes"]
    idx = onds.index

    # Get VIX
    vix_col = None
    for col in ["^VIX", "VIX"]:
        if col in closes.columns:
            vix_col = col
            break

    if vix_col is None:
        print("  WARNING: VIX not found in closes. Using RV-only features.")
        vix_series = pd.Series(np.nan, index=idx)
    else:
        vix_series = closes[vix_col].reindex(idx)

    # ONDS returns
    onds_ret = onds["Close"].pct_change()

    features = pd.DataFrame(index=idx)

    # --- 1. Variance Risk Premium (VRP) ---
    # VIX^2 / 252 gives daily implied variance
    # Compare to realized variance (20-day rolling)
    rv_20d = onds_ret.rolling(20).std() * np.sqrt(252)
    iv_daily_var = (vix_series / 100) ** 2 / 252  # VIX is in % annualized
    rv_daily_var = rv_20d ** 2 / 252
    features["vrp"] = (iv_daily_var - rv_daily_var).shift(1)

    # --- 2. IV Rank (60d rolling percentile of VIX) ---
    features["iv_rank_60d"] = vix_series.rolling(60).apply(
        lambda x: stats.percentileofscore(x[:-1], x.iloc[-1]) / 100.0
        if len(x) > 1 else 0.5,
        raw=False
    ).shift(1)

    # --- 3. VIX Term Proxy (VIX - VIX_MA5) ---
    features["vix_term"] = (vix_series - vix_series.rolling(5).mean()).shift(1)

    # --- 4. Skew Proxy (rolling asymmetry of ONDS returns) ---
    features["skew_proxy"] = onds_ret.rolling(20).apply(
        lambda x: stats.skew(x) if len(x) > 5 else 0, raw=True
    ).shift(1)

    # --- 5. VIX Regime (high-vol flag) ---
    vix_80pct = vix_series.rolling(60).quantile(0.80)
    features["vix_regime"] = (vix_series > vix_80pct).astype(float).shift(1)

    # --- 6. RV Ratio (short-term / long-term vol) ---
    rv_5d = onds_ret.rolling(5).std() * np.sqrt(252)
    rv_20d_safe = rv_20d.replace(0, np.nan)
    features["rv_ratio"] = (rv_5d / rv_20d_safe).shift(1)

    # --- 7. VIX-ONDS Correlation (rolling 20d) ---
    vix_ret = vix_series.pct_change()
    features["vix_onds_corr"] = vix_ret.rolling(20).corr(onds_ret).shift(1)

    # Clean
    features = features.replace([np.inf, -np.inf], np.nan)

    return features


def plot_iv_features(features: pd.DataFrame) -> None:
    """Plot time series of all 7 IV proxy features."""
    cols = [c for c in features.columns if not c.startswith("_")]
    n = len(cols)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.5 * n), sharex=True)

    titles = {
        "vrp": "Variance Risk Premium (VIX^2 - RV^2)",
        "iv_rank_60d": "IV Rank (60-day percentile of VIX)",
        "vix_term": "VIX Term Structure Proxy (VIX - MA5)",
        "skew_proxy": "Return Skew Proxy (20-day rolling skewness)",
        "vix_regime": "VIX Regime Flag (1 = high-vol)",
        "rv_ratio": "RV Ratio (5d / 20d vol)",
        "vix_onds_corr": "VIX-ONDS Rolling Correlation (20d)",
    }

    for ax, col in zip(axes, cols):
        data = features[col].dropna()
        ax.plot(data.index, data.values, linewidth=1.0, color="#2196F3")
        ax.set_title(titles.get(col, col), fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.4)

    axes[-1].set_xlabel("Date")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.tight_layout()

    path = FIGURES_DIR / "iv_proxy_features.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# =====================================================================
# SECTION C: IV-BASED TRADING STRATEGIES
# =====================================================================

def strategy_vrp_mean_reversion(data: dict,
                                 iv_features: pd.DataFrame) -> pd.Series:
    """
    VRP Mean Reversion:
    High VRP (fear) + ONDS oversold -> long
    Low VRP (complacency) + ONDS overbought -> short

    Logic: When options are expensive relative to realized vol (high VRP),
    fear is elevated. If ONDS is also oversold, expect a bounce.
    """
    onds = data["onds"]
    onds_ret = onds["Close"].pct_change()
    signal = pd.Series(0.0, index=onds.index)

    vrp = iv_features["vrp"]
    vrp_z = (vrp - vrp.rolling(40).mean()) / (vrp.rolling(40).std() + 1e-8)

    # ONDS oversold/overbought (5-day return z-score)
    ret_5d = onds_ret.rolling(5).sum()
    ret_z = (ret_5d - ret_5d.rolling(40).mean()) / (ret_5d.rolling(40).std() + 1e-8)

    # Shift to avoid look-ahead
    vrp_z = vrp_z.shift(1)
    ret_z = ret_z.shift(1)

    # High VRP + oversold -> long
    long_cond = (vrp_z > 1.0) & (ret_z < -1.0)
    short_cond = (vrp_z < -1.0) & (ret_z > 1.0)

    signal[long_cond] = 1.0
    signal[short_cond] = -1.0

    return signal.clip(-1, 1)


def strategy_iv_regime_conditional(data: dict,
                                    iv_features: pd.DataFrame) -> pd.Series:
    """
    IV Regime Conditional:
    Low-VIX regime: use momentum (trend-following)
    High-VIX regime: use mean-reversion (fade fear spikes)

    Logic: Different regimes favor different strategies. Low-vol regimes
    tend to trend; high-vol regimes tend to mean-revert.
    """
    onds = data["onds"]
    close = onds["Close"]
    onds_ret = close.pct_change()
    signal = pd.Series(0.0, index=onds.index)

    vix_regime = iv_features["vix_regime"].shift(1)

    # Momentum signal (SMA crossover)
    sma_fast = close.rolling(10).mean()
    sma_slow = close.rolling(30).mean()
    momentum_sig = pd.Series(0.0, index=onds.index)
    momentum_sig[sma_fast > sma_slow] = 1.0
    momentum_sig[sma_fast < sma_slow] = -1.0

    # Mean-reversion signal (fade z-score extremes)
    ret_z = (onds_ret - onds_ret.rolling(20).mean()) / (
        onds_ret.rolling(20).std() + 1e-8
    )
    mr_sig = pd.Series(0.0, index=onds.index)
    mr_sig[ret_z.shift(1) < -1.5] = 1.0
    mr_sig[ret_z.shift(1) > 1.5] = -1.0

    # Conditional: low-vol -> momentum, high-vol -> mean-reversion
    is_high_vol = vix_regime == 1.0
    signal[~is_high_vol] = momentum_sig[~is_high_vol]
    signal[is_high_vol] = mr_sig[is_high_vol]

    return signal.clip(-1, 1)


def strategy_vol_surface_composite(data: dict,
                                    iv_features: pd.DataFrame) -> pd.Series:
    """
    Vol Surface Composite:
    Weighted combination of VRP z-score + IV rank + skew proxy.

    Positive signal when:
    - VRP is high (options expensive -> fear -> reversal expected)
    - IV rank is high (vol elevated vs recent -> likely to compress)
    - Skew is negative (fat left tail -> recovery expected)
    """
    onds = data["onds"]
    signal = pd.Series(0.0, index=onds.index)

    vrp = iv_features["vrp"]
    iv_rank = iv_features["iv_rank_60d"]
    skew = iv_features["skew_proxy"]

    # Normalize to z-scores
    vrp_z = (vrp - vrp.rolling(40).mean()) / (vrp.rolling(40).std() + 1e-8)
    rank_z = (iv_rank - 0.5) * 2  # Already 0-1, center at 0
    skew_z = (skew - skew.rolling(40).mean()) / (skew.rolling(40).std() + 1e-8)

    # Composite: weighted sum (all shifted to avoid look-ahead)
    composite = (
        0.4 * vrp_z.shift(1) +
        0.3 * rank_z.shift(1) +
        0.3 * (-skew_z.shift(1))  # Negative skew is bullish (fear recovery)
    )

    # Threshold
    signal[composite > 0.8] = 1.0
    signal[composite < -0.8] = -1.0

    return signal.clip(-1, 1)


# =====================================================================
# SECTION D: INTEGRATION WITH ML PIPELINE
# =====================================================================

def run_ml_with_iv_features(data: dict, df: pd.DataFrame,
                             iv_features: pd.DataFrame) -> dict:
    """
    Test whether adding IV features improves ML model performance.

    Runs the best 2 models (RF and XGBoost) with and without IV features
    using the same leakage-free walk-forward engine from ml_models.py.
    """
    from analysis.ml_models import (
        walk_forward_predict, WalkForwardConfig, get_safe_features,
        rf_factory, xgb_factory, full_validation,
    )

    onds_close = data["onds"]["Close"]
    base_features = get_safe_features(df)

    # Add IV features to the main dataframe
    df_with_iv = df.copy()
    for col in iv_features.columns:
        df_with_iv[f"iv_{col}"] = iv_features[col]

    iv_feature_names = [f"iv_{col}" for col in iv_features.columns]
    extended_features = base_features + iv_feature_names

    config = WalkForwardConfig()
    target = "target_direction"

    results = {}
    models_to_test = {"RF": rf_factory, "XGBoost": xgb_factory}

    for model_name, factory in models_to_test.items():
        # Without IV features
        print(f"\n  {model_name} (no IV features)...")
        sig_base = walk_forward_predict(
            df, base_features, target, factory, config
        )
        bt_base = backtest(onds_close, sig_base,
                            name=f"{model_name} (base)", plot=False)

        # With IV features
        print(f"  {model_name} (+ IV features)...")
        sig_iv = walk_forward_predict(
            df_with_iv, extended_features, target, factory, config
        )
        bt_iv = backtest(onds_close, sig_iv,
                          name=f"{model_name} (+IV)", plot=False)

        results[model_name] = {
            "base_sharpe": bt_base["sharpe"],
            "iv_sharpe": bt_iv["sharpe"],
            "uplift": bt_iv["sharpe"] - bt_base["sharpe"],
            "base_return": bt_base["annual_return"],
            "iv_return": bt_iv["annual_return"],
        }

        print(f"    Base Sharpe: {bt_base['sharpe']:+.4f}")
        print(f"    +IV  Sharpe: {bt_iv['sharpe']:+.4f}")
        print(f"    Uplift:      {results[model_name]['uplift']:+.4f}")

    return results


# =====================================================================
# MAIN RUNNER
# =====================================================================

def run_iv_analysis():
    """Run complete IV surface analysis."""
    print("=" * 70)
    print("  IV SURFACE ANALYSIS")
    print("  SSVI Model + IV-Proxy Features + IV Strategies")
    print("=" * 70)

    # --- Load data ---
    print("\n[1/6] Loading data...")
    data = load_all_data()
    df = build_advanced_features(data)
    onds_close = data["onds"]["Close"]
    current_price = float(onds_close.iloc[-1])
    print(f"  ONDS current price: ${current_price:.2f}")

    # --- SSVI Surface Fitting ---
    print("\n[2/6] Fitting SSVI surface to ONDS options...")
    from analysis.options_iv import load_options_data
    options_df = load_options_data(TARGET_TICKER)

    if options_df.empty:
        print("  No options data available. Skipping SSVI fit.")
        ssvi_result = {"error": "No data"}
    else:
        print(f"  Options contracts: {len(options_df)}")
        ssvi_result = fit_ssvi_surface(options_df, current_price)

        if "error" in ssvi_result:
            print(f"  SSVI fit failed: {ssvi_result['error']}")
        else:
            print(f"  SSVI parameters:")
            print(f"    rho (skew):     {ssvi_result['params']['rho']:+.4f}")
            print(f"    eta (vol-vol):  {ssvi_result['params']['eta']:.4f}")
            print(f"    gamma (power):  {ssvi_result['params']['gamma']:.4f}")
            print(f"  Fit quality:")
            print(f"    RMSE:           {ssvi_result['rmse']:.6f}")
            print(f"    R2:             {ssvi_result['r2']:.4f}")
            print(f"    RMSE/ATM:       {ssvi_result['rmse_pct_of_atm']:.1f}%")
            plot_ssvi_surface(ssvi_result)

    # Save SSVI results
    if "error" not in ssvi_result:
        ssvi_csv = pd.DataFrame([{
            "rho": ssvi_result["params"]["rho"],
            "eta": ssvi_result["params"]["eta"],
            "gamma": ssvi_result["params"]["gamma"],
            "rmse": ssvi_result["rmse"],
            "r2": ssvi_result["r2"],
            "rmse_pct_of_atm": ssvi_result["rmse_pct_of_atm"],
            "n_contracts": ssvi_result["n_contracts"],
            "n_expirations": ssvi_result["n_expirations"],
        }])
        ssvi_csv.to_csv(REPORTS_DIR / "iv_surface_analysis.csv", index=False)
        print(f"  Saved: {REPORTS_DIR / 'iv_surface_analysis.csv'}")

    # --- Build IV Proxy Features ---
    print("\n[3/6] Building daily IV-proxy features...")
    iv_features = build_iv_proxy_features(data)
    n_valid = iv_features.dropna(how="all").shape[0]
    print(f"  Features: {len(iv_features.columns)}")
    print(f"  Valid days: {n_valid} / {len(iv_features)}")

    for col in iv_features.columns:
        valid = iv_features[col].dropna()
        if len(valid) > 0:
            print(f"    {col:20s}: mean={valid.mean():+.4f}  "
                  f"std={valid.std():.4f}  "
                  f"range=[{valid.min():.4f}, {valid.max():.4f}]")

    plot_iv_features(iv_features)

    # Save IV features
    iv_features.to_csv(DATA_PROC / "iv_proxy_features.csv")
    print(f"  Saved: {DATA_PROC / 'iv_proxy_features.csv'}")

    # --- IV-Based Strategies ---
    print("\n[4/6] Running IV-based strategies...")
    print("-" * 70)

    strategies = {
        "VRP Mean Reversion": strategy_vrp_mean_reversion(data, iv_features),
        "IV Regime Conditional": strategy_iv_regime_conditional(data, iv_features),
        "Vol Surface Composite": strategy_vol_surface_composite(data, iv_features),
    }

    # Add vol-adjusted versions
    va_strategies = {}
    for name, sig in strategies.items():
        va_strategies[f"{name} (VolAdj)"] = vol_adjusted_signal(
            sig, onds_close, target_vol=0.30
        )
    strategies.update(va_strategies)

    # Backtest all
    all_results = []
    for name, sig in strategies.items():
        n_active = (sig != 0).sum()
        if n_active < 5:
            print(f"  {name}: SKIPPED (only {n_active} active days)")
            continue

        bt = backtest(onds_close, sig, name=name, plot=False, save_fig=False)
        all_results.append(bt)
        print(f"  {name:35s} Sharpe={bt['sharpe']:+.4f}  "
              f"Return={bt['annual_return']:+.2%}  "
              f"MaxDD={bt['max_drawdown']:.2%}  "
              f"Trades={bt['n_trades']}")

    # 60/10/30 split for IV strategies
    print("\n  60/10/30 Split Results:")
    train_end, val_end = make_split(onds_close.index)
    iv_split_rows = []

    for name, sig in strategies.items():
        if (sig != 0).sum() < 5:
            continue
        train_bt = period_backtest(onds_close, sig, onds_close.index[0],
                                    train_end, name=f"{name} (Train)")
        test_bt = period_backtest(onds_close, sig, val_end,
                                   onds_close.index[-1], name=f"{name} (Test)")
        train_s = train_bt["sharpe"] if train_bt else 0
        test_s = test_bt["sharpe"] if test_bt else 0
        gap = abs(train_s - test_s)
        print(f"    {name:35s} Train={train_s:+.4f}  "
              f"Test={test_s:+.4f}  Gap={gap:.2f}")
        iv_split_rows.append({
            "Strategy": name,
            "Train Sharpe": train_s,
            "Test Sharpe": test_s,
            "Gap": gap,
        })

    # Bootstrap for top strategies
    print("\n  Bootstrap Validation (top IV strategies):")
    for name, sig in strategies.items():
        if (sig != 0).sum() < 10:
            continue
        bt = backtest(onds_close, sig, name=name, plot=False, save_fig=False)
        bs = bootstrap_sharpe(bt["daily_pnl"], n_boot=5000)
        print(f"    {name:35s} Sharpe={bs['mean']:+.4f}  "
              f"95%CI=[{bs['ci_2.5']:+.2f}, {bs['ci_97.5']:+.2f}]  "
              f"P(>0)={bs['pct_positive']:.1%}")

    # Plot equity curves
    if all_results:
        fig, ax = plt.subplots(figsize=(14, 7))
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
        for r, c in zip(all_results, colors):
            ax.plot(r["cum_return"], label=f"{r['name']} (S={r['sharpe']:.2f})",
                    color=c, linewidth=1.2)
        bm = all_results[0]["cum_benchmark"]
        ax.plot(bm, label="Buy & Hold", color="gray", linestyle="--", alpha=0.6)
        ax.set_title("IV-Based Strategies -- Equity Curves")
        ax.set_ylabel("Cumulative Return")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = FIGURES_DIR / "iv_strategy_equity.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  Saved: {path}")

    # Save strategy results
    if iv_split_rows:
        iv_csv = pd.DataFrame(iv_split_rows)
        iv_csv.to_csv(REPORTS_DIR / "iv_strategy_results.csv", index=False)
        print(f"  Saved: {REPORTS_DIR / 'iv_strategy_results.csv'}")

    # --- ML Integration ---
    print("\n[5/6] Testing IV feature uplift in ML models...")
    print("-" * 70)
    ml_iv_results = run_ml_with_iv_features(data, df, iv_features)

    print("\n  IV Feature Uplift Summary:")
    print(f"  {'Model':<15s} {'Base':>10s} {'+ IV':>10s} {'Uplift':>10s}")
    print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10}")
    for model, res in ml_iv_results.items():
        print(f"  {model:<15s} {res['base_sharpe']:>+10.4f} "
              f"{res['iv_sharpe']:>+10.4f} {res['uplift']:>+10.4f}")

    # --- VRP Sign Check ---
    print("\n[6/6] Sanity checks...")
    print("-" * 70)

    # Check VRP -> next-day ONDS return correlation
    onds_ret = onds_close.pct_change().shift(-1)  # next-day return
    vrp = iv_features["vrp"]
    valid = pd.DataFrame({"vrp": vrp, "next_ret": onds_ret}).dropna()
    if len(valid) > 20:
        corr = valid["vrp"].corr(valid["next_ret"])
        print(f"  VRP -> next-day ONDS return correlation: {corr:+.4f}")
        sign = "EXPECTED (positive)" if corr > 0 else "UNEXPECTED (negative)"
        print(f"  Sign: {sign}")
        print(f"  (Positive VRP = fear premium -> expected positive next-day return)")

    # Check IV features have non-zero values for all days
    for col in iv_features.columns:
        n_valid = iv_features[col].dropna().shape[0]
        n_total = len(iv_features)
        pct = n_valid / n_total * 100
        status = "OK" if pct > 80 else "LOW COVERAGE"
        print(f"  {col:20s}: {n_valid}/{n_total} valid ({pct:.0f}%) [{status}]")

    print("\n" + "=" * 70)
    print("  IV SURFACE ANALYSIS COMPLETE")
    print("=" * 70)

    return {
        "ssvi": ssvi_result,
        "iv_features": iv_features,
        "strategies": strategies,
        "ml_uplift": ml_iv_results,
    }


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    results = run_iv_analysis()
