"""
Alpha Decay Detection & Strategy Health Dashboard
====================================================
Rolling IC, cumulative alpha slope, OU half-life,
CUSUM structural break, and traffic-light health scoring.
"""
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
import sys

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FIGURES_DIR, REPORTS_DIR, BACKTEST
from backtests.engine import backtest, compute_returns
from analysis.advanced_research import (
    load_all_data, build_advanced_features, select_features,
    vol_adjusted_signal,
    strategy_peer_leadlag, strategy_market_contrarian,
    strategy_momentum_with_stop, strategy_mean_reversion,
    strategy_multi_timeframe_momentum, strategy_combined_best,
    strategy_ml_signal, strategy_adaptive_ensemble,
    strategy_regime_conditional, strategy_dix_enhanced,
    strategy_volatility_breakout, strategy_gap_fade,
    strategy_volume_spike, strategy_rsi_divergence,
    strategy_bollinger_mean_reversion,
)


# =====================================================================
# ROLLING INFORMATION COEFFICIENT
# =====================================================================

def rolling_ic(signal: pd.Series, forward_returns: pd.Series,
               window: int = 40) -> pd.Series:
    """
    Rolling Spearman rank-correlation between signal and next-day returns.
    Measures whether the signal still ranks days correctly.
    """
    common = signal.index.intersection(forward_returns.index)
    sig = signal.reindex(common)
    fwd = forward_returns.reindex(common)
    ic = pd.Series(np.nan, index=common)
    for i in range(window, len(common)):
        s = sig.iloc[i - window:i]
        f = fwd.iloc[i - window:i]
        mask = s.notna() & f.notna() & (s != 0)
        if mask.sum() >= 10:
            rho, _ = stats.spearmanr(s[mask], f[mask])
            ic.iloc[i] = rho
    return ic


def ic_decay_by_lag(signal: pd.Series, prices: pd.Series,
                    lags=(1, 2, 3, 5, 10, 20)) -> dict:
    """
    IC at different forward horizons.
    Shows how fast the signal's predictive power decays with holding period.
    """
    ret = prices.pct_change()
    results = {}
    for lag in lags:
        fwd = ret.shift(-lag)
        common = signal.index.intersection(fwd.dropna().index)
        s = signal.reindex(common).dropna()
        f = fwd.reindex(s.index).dropna()
        common2 = s.index.intersection(f.index)
        s, f = s.loc[common2], f.loc[common2]
        mask = s != 0
        if mask.sum() >= 20:
            rho, pval = stats.spearmanr(s[mask], f[mask])
            results[lag] = {"ic": rho, "p_value": pval, "n": int(mask.sum())}
        else:
            results[lag] = {"ic": np.nan, "p_value": np.nan, "n": int(mask.sum())}
    return results


# =====================================================================
# CUMULATIVE ALPHA
# =====================================================================

def cumulative_alpha(strat_returns: pd.Series,
                     bench_returns: pd.Series) -> pd.Series:
    """Cumulative excess return (strategy - benchmark), compounded."""
    common = strat_returns.index.intersection(bench_returns.index)
    excess = strat_returns.reindex(common) - bench_returns.reindex(common)
    return (1 + excess).cumprod() - 1


def alpha_slope(cum_alpha: pd.Series, window: int = 40) -> pd.Series:
    """
    Rolling OLS slope of cumulative alpha.
    Positive = alpha still accruing; negative = alpha decaying.
    """
    slope = pd.Series(np.nan, index=cum_alpha.index)
    x = np.arange(window, dtype=float)
    for i in range(window, len(cum_alpha)):
        y = cum_alpha.iloc[i - window:i].values
        if np.any(np.isnan(y)):
            continue
        m, _, _, _, _ = stats.linregress(x, y)
        slope.iloc[i] = m
    return slope


# =====================================================================
# ORNSTEIN-UHLENBECK HALF-LIFE
# =====================================================================

def half_life_OU(series: pd.Series) -> float:
    """
    Fit OU process to cumulative alpha:  dx = theta*(mu - x)*dt + sigma*dW
    Half-life = ln(2) / theta.
    Returns half-life in days (np.inf if non-mean-reverting).
    """
    y = series.dropna().values
    if len(y) < 20:
        return np.inf
    dy = np.diff(y)
    y_lag = y[:-1]
    # OLS: dy = a + b*y_lag  →  theta = -b
    slope, intercept, _, p_val, _ = stats.linregress(y_lag, dy)
    if slope >= 0:          # not mean-reverting
        return np.inf
    theta = -slope
    hl = np.log(2) / theta
    return hl


# =====================================================================
# STRUCTURAL BREAK — CUSUM
# =====================================================================

def cusum_test(strat_returns: pd.Series,
               bench_returns: pd.Series) -> dict:
    """
    CUSUM test on excess returns.
    Detects a structural shift in strategy performance.
    Returns dict with cusum series, max deviation, and critical value.
    """
    common = strat_returns.index.intersection(bench_returns.index)
    excess = strat_returns.reindex(common) - bench_returns.reindex(common)
    excess = excess.dropna()
    n = len(excess)
    if n < 20:
        return {"cusum": pd.Series(dtype=float), "max_dev": 0,
                "critical_5pct": 0, "break_detected": False}

    mu = excess.mean()
    sigma = excess.std()
    if sigma == 0:
        sigma = 1e-8
    cusum = ((excess - mu) / sigma).cumsum()

    # Brownian bridge normalisation (divide by sqrt(n) for correct scale)
    t = np.arange(1, n + 1) / n
    bridge = (cusum.values - t * cusum.values[-1]) / np.sqrt(n)
    max_dev = np.max(np.abs(bridge))

    # Approximate 5% critical value for sup|W0(t)| (Brownian bridge)
    critical = 1.358

    return {
        "cusum": pd.Series(bridge, index=excess.index),
        "max_dev": max_dev,
        "critical_5pct": critical,
        "break_detected": max_dev > critical,
    }


def chow_test_scan(strat_returns: pd.Series, bench_returns: pd.Series,
                   min_segment: int = 40) -> dict:
    """
    Scan for the single most likely structural breakpoint.
    Returns date and F-statistic of the best split.
    """
    common = strat_returns.index.intersection(bench_returns.index)
    excess = (strat_returns.reindex(common) - bench_returns.reindex(common)).dropna()
    n = len(excess)
    if n < 2 * min_segment:
        return {"breakpoint": None, "f_stat": 0, "p_value": 1.0}

    vals = excess.values
    best_f = 0
    best_idx = min_segment

    rss_full = np.sum((vals - vals.mean()) ** 2)

    for k in range(min_segment, n - min_segment):
        seg1 = vals[:k]
        seg2 = vals[k:]
        rss1 = np.sum((seg1 - seg1.mean()) ** 2)
        rss2 = np.sum((seg2 - seg2.mean()) ** 2)
        rss_reduced = rss1 + rss2
        # F = ((RSS_full - RSS_reduced) / p) / (RSS_reduced / (n - 2*p))
        p = 1  # 1 parameter (mean)
        denom = rss_reduced / (n - 2 * p) if rss_reduced > 0 else 1e-8
        f_stat = ((rss_full - rss_reduced) / p) / denom
        if f_stat > best_f:
            best_f = f_stat
            best_idx = k

    from scipy.stats import f as f_dist
    p_value = 1 - f_dist.cdf(best_f, 1, n - 2)

    return {
        "breakpoint": excess.index[best_idx],
        "f_stat": best_f,
        "p_value": p_value,
    }


# =====================================================================
# STRATEGY HEALTH DASHBOARD
# =====================================================================

def _traffic_light(value, thresholds):
    """Return GREEN / YELLOW / RED based on (green_min, yellow_min)."""
    green_min, yellow_min = thresholds
    if value >= green_min:
        return "GREEN"
    elif value >= yellow_min:
        return "YELLOW"
    return "RED"


def strategy_health(bt_result: dict, signal: pd.Series,
                    prices: pd.Series) -> dict:
    """
    Compute health metrics for a single strategy.
    Returns dict with metric values and traffic-light colours.
    """
    daily = bt_result["daily_pnl"]
    cum   = bt_result["cum_return"]
    bench_ret = compute_returns(prices).reindex(daily.index)

    # ── Rolling Sharpe (60d) ─────────────────────────────────────
    roll_sharpe_60 = (
        daily.rolling(60).mean() / daily.rolling(60).std()
    ) * np.sqrt(252)
    recent_sharpe = roll_sharpe_60.iloc[-20:].mean() if len(roll_sharpe_60) >= 20 else 0

    # ── Drawdown severity ────────────────────────────────────────
    peak = cum.cummax()
    dd = (cum - peak) / peak
    current_dd = dd.iloc[-1]
    max_dd = dd.min()

    # ── Hit rate (recent 60d) ────────────────────────────────────
    active = daily[daily != 0]
    recent_active = active.iloc[-60:]
    recent_hit = (recent_active > 0).mean() if len(recent_active) > 0 else 0

    # ── Rolling IC (40d) ─────────────────────────────────────────
    fwd_ret = prices.pct_change().shift(-1)
    ric = rolling_ic(signal, fwd_ret, window=40)
    recent_ic = ric.iloc[-20:].mean() if len(ric.dropna()) >= 20 else 0

    # ── Profit factor (recent 60d) ───────────────────────────────
    gains = recent_active[recent_active > 0].sum() if len(recent_active) > 0 else 0
    losses = -recent_active[recent_active < 0].sum() if len(recent_active) > 0 else 1e-8
    pf = gains / max(losses, 1e-8)

    # ── Alpha slope ──────────────────────────────────────────────
    cum_a = cumulative_alpha(daily, bench_ret)
    aslope = alpha_slope(cum_a, window=40)
    recent_slope = aslope.iloc[-10:].mean() if len(aslope.dropna()) >= 10 else 0

    # ── Traffic lights ───────────────────────────────────────────
    lights = {
        "sharpe":  _traffic_light(recent_sharpe, (0.5, -0.5)),
        "dd":      _traffic_light(-abs(current_dd), (-0.10, -0.25)),
        "hit":     _traffic_light(recent_hit, (0.52, 0.45)),
        "ic":      _traffic_light(recent_ic, (0.05, -0.02)),
        "pf":      _traffic_light(pf, (1.2, 0.8)),
        "slope":   _traffic_light(recent_slope, (0.0005, -0.0005)),
    }

    n_green  = sum(1 for v in lights.values() if v == "GREEN")
    n_red    = sum(1 for v in lights.values() if v == "RED")
    if n_red >= 3:
        verdict = "RECALIBRATE"
    elif n_green >= 4:
        verdict = "HEALTHY"
    else:
        verdict = "MONITOR"

    return {
        "recent_sharpe": round(recent_sharpe, 3),
        "current_dd": round(current_dd, 4),
        "max_dd": round(max_dd, 4),
        "recent_hit": round(recent_hit, 4),
        "recent_ic": round(recent_ic, 4),
        "profit_factor": round(pf, 3),
        "alpha_slope": round(recent_slope, 6),
        "lights": lights,
        "verdict": verdict,
    }


# =====================================================================
# VISUALISATION
# =====================================================================

def plot_alpha_decay_dashboard(strategies_health, save=True):
    """Multi-panel dashboard: rolling Sharpe, IC, alpha slope per strategy."""
    n = len(strategies_health)
    if n == 0:
        return
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    cmap = plt.cm.tab10(np.linspace(0, 1, max(n, 1)))

    for i, (name, info) in enumerate(strategies_health.items()):
        c = cmap[i % len(cmap)]
        if "rolling_sharpe" in info:
            axes[0].plot(info["rolling_sharpe"], label=name, color=c, linewidth=1)
        if "rolling_ic" in info:
            axes[1].plot(info["rolling_ic"], label=name, color=c, linewidth=1)
        if "alpha_slope_series" in info:
            axes[2].plot(info["alpha_slope_series"], label=name, color=c, linewidth=1)

    axes[0].axhline(0, color="gray", linewidth=0.5, linestyle="--")
    axes[0].set_ylabel("Rolling Sharpe (60d)")
    axes[0].set_title("Alpha Decay Dashboard")
    axes[0].legend(loc="upper left", fontsize=7, ncol=2)
    axes[0].grid(True, alpha=0.3)

    axes[1].axhline(0, color="gray", linewidth=0.5, linestyle="--")
    axes[1].set_ylabel("Rolling IC (40d)")
    axes[1].grid(True, alpha=0.3)

    axes[2].axhline(0, color="gray", linewidth=0.5, linestyle="--")
    axes[2].set_ylabel("Alpha Slope (40d)")
    axes[2].set_xlabel("Date")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        path = FIGURES_DIR / "alpha_decay_dashboard.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


def plot_cumulative_alpha(strategies_cum_alpha, save=True):
    """Overlay cumulative alpha curves."""
    fig, ax = plt.subplots(figsize=(14, 7))
    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(strategies_cum_alpha), 1)))

    for i, (name, ca) in enumerate(strategies_cum_alpha.items()):
        ax.plot(ca, label=name, color=cmap[i % len(cmap)], linewidth=1.2)

    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_title("Cumulative Alpha (Strategy - Buy & Hold)")
    ax.set_ylabel("Cumulative Excess Return")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        path = FIGURES_DIR / "cumulative_alpha.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


# =====================================================================
# MAIN
# =====================================================================

def run_alpha_decay():
    print("=" * 70)
    print("  ALPHA DECAY DETECTION & STRATEGY HEALTH")
    print("=" * 70)

    # ── Load ──────────────────────────────────────────────────────
    print("\n[1/5] Loading data...")
    data = load_all_data()

    print("\n[2/5] Building features (for ML strategy)...")
    df = build_advanced_features(data)
    leakage = {"ret_open_close", "ret_high_low", "ret_close_high", "ret_close_low",
               "gap", "volume_change", "volume_ma_ratio", "dollar_volume",
               "price_range_pct", "ret_1d", "dow", "is_monday", "is_friday"}
    all_feat = [c for c in df.columns
                if not c.startswith("target_") and c not in leakage]
    feature_cols = select_features(df, "target_direction", n_top=15,
                                   allowed_cols=all_feat)

    onds_close = data["onds"]["Close"]
    bench_ret = compute_returns(onds_close)

    # ── Generate all signals ──────────────────────────────────────
    print("\n[3/5] Generating strategy signals & backtesting...")
    strategy_specs = [
        ("Peer Lead-Lag",        strategy_peer_leadlag(data)),
        ("Market Contrarian",    strategy_market_contrarian(data)),
        ("Momentum+Stop",        strategy_momentum_with_stop(data)),
        ("Mean Reversion",       strategy_mean_reversion(data)),
        ("Multi-TF Momentum",    strategy_multi_timeframe_momentum(data)),
        ("Combined Best",        strategy_combined_best(data, df)),
        ("Regime Conditional",   strategy_regime_conditional(data)),
        ("DIX Enhanced",         strategy_dix_enhanced(data)),
        ("Vol Breakout",         strategy_volatility_breakout(data)),
        ("Gap Fade",             strategy_gap_fade(data)),
        ("Adaptive Ensemble",    strategy_adaptive_ensemble(data, df)),
        ("ML Direction",         strategy_ml_signal(df, feature_cols, "target_direction")),
    ]

    # Backtest all
    bt_results = {}
    signals = {}
    for name, sig in strategy_specs:
        try:
            r = backtest(onds_close, sig, name=name, plot=False, save_fig=False)
            bt_results[name] = r
            signals[name] = sig
        except Exception as e:
            print(f"  {name}: backtest failed — {e}")

    # ── Decay analysis ────────────────────────────────────────────
    print(f"\n[4/5] Computing decay metrics for {len(bt_results)} strategies...")
    fwd_ret = onds_close.pct_change().shift(-1)
    decay_rows = []
    strategies_health = {}
    strategies_cum_alpha = {}

    for name, r in bt_results.items():
        sig = signals[name]
        daily = r["daily_pnl"]

        # Rolling IC
        ric = rolling_ic(sig, fwd_ret, window=40)

        # IC decay by lag
        ic_lag = ic_decay_by_lag(sig, onds_close)

        # Cumulative alpha
        ca = cumulative_alpha(daily, bench_ret.reindex(daily.index))
        strategies_cum_alpha[name] = ca

        # Alpha slope
        aslope = alpha_slope(ca, window=40)

        # OU half-life
        hl = half_life_OU(ca)

        # CUSUM
        cusum_res = cusum_test(daily, bench_ret.reindex(daily.index))

        # Chow breakpoint
        chow_res = chow_test_scan(daily, bench_ret.reindex(daily.index))

        # Health
        health = strategy_health(r, sig, onds_close)

        # Rolling Sharpe for dashboard
        roll_s = (daily.rolling(60).mean() / daily.rolling(60).std()) * np.sqrt(252)

        strategies_health[name] = {
            **health,
            "rolling_sharpe": roll_s,
            "rolling_ic": ric,
            "alpha_slope_series": aslope,
        }

        decay_rows.append({
            "strategy":          name,
            "full_sharpe":       r["sharpe"],
            "recent_sharpe":     health["recent_sharpe"],
            "recent_ic":         health["recent_ic"],
            "ic_lag1":           ic_lag[1]["ic"] if 1 in ic_lag else np.nan,
            "ic_lag5":           ic_lag[5]["ic"] if 5 in ic_lag else np.nan,
            "ic_lag10":          ic_lag[10]["ic"] if 10 in ic_lag else np.nan,
            "ou_half_life":      round(hl, 1) if np.isfinite(hl) else "Inf",
            "alpha_slope":       health["alpha_slope"],
            "cusum_break":       cusum_res["break_detected"],
            "cusum_max_dev":     round(cusum_res["max_dev"], 3),
            "chow_breakpoint":   str(chow_res["breakpoint"].date())
                                 if chow_res["breakpoint"] is not None else "None",
            "chow_p_value":      round(chow_res["p_value"], 4),
            "profit_factor":     health["profit_factor"],
            "verdict":           health["verdict"],
        })

    # ── Save & plot ───────────────────────────────────────────────
    print("\n[5/5] Saving reports & figures...")
    decay_df = pd.DataFrame(decay_rows)
    decay_df.to_csv(REPORTS_DIR / "alpha_decay_report.csv", index=False)
    print(f"  Saved: {REPORTS_DIR / 'alpha_decay_report.csv'}")

    plot_alpha_decay_dashboard(strategies_health)
    plot_cumulative_alpha(strategies_cum_alpha)

    # ── Print dashboard ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  STRATEGY HEALTH DASHBOARD")
    print("=" * 70)
    print(f"  {'Strategy':<22s} {'Verdict':<14s} {'Sharpe':>7s} {'IC':>7s} "
          f"{'Hit':>6s} {'PF':>6s} {'Slope':>9s} {'OU HL':>7s}")
    print("  " + "-" * 80)
    for row in sorted(decay_rows, key=lambda x: x["full_sharpe"], reverse=True):
        h = strategies_health[row["strategy"]]
        lights = h["lights"]
        # Colour-code verdict
        v = row["verdict"]
        hl_str = str(row["ou_half_life"])
        print(f"  {row['strategy']:<22s} {v:<14s} "
              f"{row['recent_sharpe']:>+7.3f} "
              f"{row['recent_ic']:>7.4f} "
              f"{h['recent_hit']:>6.2%} "
              f"{h['profit_factor']:>6.2f} "
              f"{row['alpha_slope']:>+9.6f} "
              f"{hl_str:>7s}")

    # Light summary
    print(f"\n  Light details (G=GREEN, Y=YELLOW, R=RED):")
    for name in bt_results:
        h = strategies_health[name]
        codes = "".join(v[0] for v in h["lights"].values())
        print(f"    {name:<22s} [{codes}]  "
              f"sharpe={h['lights']['sharpe']} "
              f"dd={h['lights']['dd']} "
              f"hit={h['lights']['hit']} "
              f"ic={h['lights']['ic']} "
              f"pf={h['lights']['pf']} "
              f"slope={h['lights']['slope']}")

    # CUSUM alerts
    breaks = [r for r in decay_rows if r["cusum_break"]]
    if breaks:
        print(f"\n  CUSUM STRUCTURAL BREAK ALERTS:")
        for b in breaks:
            print(f"    {b['strategy']}: max_dev={b['cusum_max_dev']:.3f} > 1.358  "
                  f"-> regime shift detected")
    else:
        print(f"\n  No CUSUM structural breaks detected at 5% level.")

    return decay_df, strategies_health


if __name__ == "__main__":
    run_alpha_decay()
