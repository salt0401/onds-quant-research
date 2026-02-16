"""
Parameter Optimization with 60/10/30 Split
============================================
Grid-search strategy parameters on Train (60%) / Validation (10%) / Test (30%).
Selects params that generalise across both in-sample windows before final evaluation.
"""
import numpy as np
import pandas as pd
from itertools import product
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
import sys

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FIGURES_DIR, REPORTS_DIR, BACKTEST
from backtests.engine import backtest
from analysis.advanced_research import (
    load_all_data, build_advanced_features, select_features,
    vol_adjusted_signal,
)


# =====================================================================
# DATA SPLIT (60 / 10 / 30)
# =====================================================================

def make_split(index: pd.DatetimeIndex):
    """Return (train_end, val_end) date boundaries for 60/10/30 split."""
    n = len(index)
    train_end = index[int(n * 0.60) - 1]
    val_end   = index[int(n * 0.70) - 1]
    return train_end, val_end


def period_backtest(prices, signal, start, end, name=""):
    """Backtest on a date-sliced sub-period. Returns result dict or None."""
    mask = (prices.index >= start) & (prices.index <= end)
    p = prices.loc[mask]
    s = signal.reindex(p.index).fillna(0)
    if len(p) < 10:
        return None
    return backtest(p, s, name=name, plot=False, save_fig=False)


# =====================================================================
# PARAMETERISED STRATEGY WRAPPERS
# =====================================================================

def momentum_stop_params(data, params):
    """Momentum + trailing stop with tuneable SMA windows and stop levels."""
    onds = data["onds"]
    close = onds["Close"]
    sma_short = params.get("sma_short", 10)
    sma_long  = params.get("sma_long", 50)
    trailing  = params.get("trailing_stop", 0.10)
    max_loss  = params.get("max_loss", 0.15)

    sma_s = close.rolling(sma_short).mean()
    sma_l = close.rolling(sma_long).mean()
    sig = pd.Series(0.0, index=onds.index)

    position = 0
    entry_price = 0.0
    peak_price = 0.0
    warmup = max(sma_short, sma_long)

    for i in range(warmup, len(sig)):
        price = close.iloc[i]
        if position == 0:
            if price > sma_s.iloc[i] and sma_s.iloc[i] > sma_l.iloc[i]:
                position = 1
                entry_price = price
                peak_price = price
        elif position == 1:
            peak_price = max(peak_price, price)
            if price < sma_s.iloc[i]:
                position = 0
            elif price < peak_price * (1 - trailing):
                position = 0
            elif price < entry_price * (1 - max_loss):
                position = 0
        sig.iloc[i] = position
    return sig


def market_contrarian_params(data, params):
    """Market contrarian with tuneable z-threshold and lookback."""
    onds = data["onds"]
    closes = data["closes"].reindex(onds.index)
    z_thresh = params.get("z_threshold", 0.75)
    lookback = params.get("lookback", 20)
    sig = pd.Series(0.0, index=onds.index)

    if "QQQ" in closes.columns:
        nr = closes["QQQ"].pct_change()
        nz = (nr - nr.rolling(lookback).mean()) / nr.rolling(lookback).std()
        sig += np.where(nz > z_thresh, -0.5, np.where(nz < -z_thresh, 0.5, 0))

    if "SPY" in closes.columns:
        sr = closes["SPY"].pct_change()
        sz = (sr - sr.rolling(lookback).mean()) / sr.rolling(lookback).std()
        sig += np.where(sz > z_thresh, -0.3, np.where(sz < -z_thresh, 0.3, 0))

    return pd.Series(sig, index=onds.index).clip(-1, 1)


def ml_direction_params(df, feature_cols_full, params):
    """Walk-forward ML direction with tuneable hyperparameters."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    n_feat     = params.get("n_features", 15)
    prob_thr   = params.get("prob_threshold", 0.58)
    max_depth  = params.get("max_depth", 4)
    n_est      = params.get("n_estimators", 150)

    feature_cols = feature_cols_full[:n_feat]
    target_col = "target_direction"
    clean = df[feature_cols + [target_col]].replace([np.inf, -np.inf], np.nan).dropna()
    X = clean[feature_cols]
    y = clean[target_col]

    scaler = StandardScaler()
    signal = pd.Series(0.0, index=clean.index)
    min_train = 60

    for i in range(min_train, len(X)):
        X_train = scaler.fit_transform(X.iloc[:i])
        X_test  = scaler.transform(X.iloc[i:i+1])
        y_train = y.iloc[:i]
        rf = RandomForestClassifier(
            n_estimators=n_est, max_depth=max_depth,
            min_samples_leaf=10, random_state=42,
        )
        rf.fit(X_train, y_train)
        prob = rf.predict_proba(X_test)[0]
        if len(prob) == 2:
            p_up = prob[1]
            if p_up > prob_thr:
                signal.iloc[i] = 1.0
            elif p_up < (1 - prob_thr):
                signal.iloc[i] = -1.0
    return signal


def peer_leadlag_params(data, params):
    """Peer lead-lag with tuneable z-threshold and lookback."""
    onds = data["onds"]
    closes = data["closes"].reindex(onds.index)
    z_thresh = params.get("z_threshold", 0.5)
    lookback = params.get("lookback", 20)
    sig = pd.Series(0.0, index=onds.index)

    if "JOBY" in closes.columns:
        jr = closes["JOBY"].pct_change()
        jl = jr.shift(1)
        jz = (jl - jl.rolling(lookback).mean()) / jl.rolling(lookback).std()
        sig += np.where(jz < -z_thresh, 0.5, np.where(jz > z_thresh, -0.5, 0))

    if "RCAT" in closes.columns:
        rr = closes["RCAT"].pct_change()
        rl = rr.shift(2)
        rz = (rl - rl.rolling(lookback).mean()) / rl.rolling(lookback).std()
        sig += np.where(rz < -z_thresh, 0.3, np.where(rz > z_thresh, -0.3, 0))

    return pd.Series(sig, index=onds.index).clip(-1, 1)


def multi_tf_momentum_params(data, params):
    """Multi-timeframe momentum with tuneable period tuple and strength divisor."""
    onds = data["onds"]
    close = onds["Close"]
    periods = params.get("periods", (5, 10, 20))
    strength_div = params.get("strength_div", 0.15)
    sig = pd.Series(0.0, index=onds.index)

    p1, p2, p3 = periods
    m1 = close.pct_change(p1)
    m2 = close.pct_change(p2)
    m3 = close.pct_change(p3)

    warmup = max(periods)
    for i in range(warmup, len(sig)):
        v1, v2, v3 = m1.iloc[i], m2.iloc[i], m3.iloc[i]
        if pd.isna(v1) or pd.isna(v2) or pd.isna(v3):
            continue
        if v1 > 0 and v2 > 0 and v3 > 0:
            sig.iloc[i] = min(1.0, (v1 + v2 + v3) / strength_div)
        elif v1 < 0 and v2 < 0 and v3 < 0:
            sig.iloc[i] = -min(1.0, abs(v1 + v2 + v3) / strength_div)

    return sig.clip(-1, 1)


def mean_reversion_params(data, params):
    """Mean reversion with tuneable z-threshold and lookback."""
    onds = data["onds"]
    ret = onds["Close"].pct_change()
    z_thresh = params.get("z_threshold", 2.0)
    lookback = params.get("lookback", 20)
    sig = pd.Series(0.0, index=onds.index)

    ret_z = (ret - ret.rolling(lookback).mean()) / ret.rolling(lookback).std()

    for i in range(lookback, len(sig)):
        z = ret_z.iloc[i]
        if z > z_thresh:
            sig.iloc[i] = -1.0
        elif z > z_thresh / 2:
            sig.iloc[i] = -0.5
        elif z < -z_thresh:
            sig.iloc[i] = 1.0
        elif z < -z_thresh / 2:
            sig.iloc[i] = 0.5
    return sig


def combined_best_params(data, params):
    """Weighted combination with tuneable weights for sub-signals."""
    w_peer    = params.get("w_peer", 0.3)
    w_contra  = params.get("w_contra", 0.3)
    w_mtf     = params.get("w_mtf", 0.2)
    w_mr      = params.get("w_mr", 0.2)

    from analysis.advanced_research import (
        strategy_peer_leadlag, strategy_market_contrarian,
        strategy_multi_timeframe_momentum, strategy_mean_reversion,
    )
    sigs = (
        strategy_peer_leadlag(data) * w_peer
        + strategy_market_contrarian(data) * w_contra
        + strategy_multi_timeframe_momentum(data) * w_mtf
        + strategy_mean_reversion(data) * w_mr
    )
    return sigs.clip(-1, 1)


# =====================================================================
# PARAMETER GRIDS
# =====================================================================

GRIDS = {
    "Momentum+Stop": {
        "sma_short":     [5, 10, 15],
        "sma_long":      [30, 50, 70],
        "trailing_stop": [0.08, 0.10, 0.15],
        "max_loss":      [0.10, 0.15, 0.20],
    },
    "Market Contrarian": {
        "z_threshold": [0.5, 0.75, 1.0, 1.25],
        "lookback":    [10, 20, 30],
    },
    "ML Direction": {
        "n_features":      [10, 15, 20],
        "prob_threshold":  [0.54, 0.56, 0.58, 0.60],
        "max_depth":       [3, 4, 5],
    },
    "Peer Lead-Lag": {
        "z_threshold": [0.3, 0.5, 0.75, 1.0],
        "lookback":    [10, 20, 30],
    },
    "Multi-TF Momentum": {
        "periods":      [(3, 7, 14), (5, 10, 20), (5, 10, 30), (10, 20, 50)],
        "strength_div": [0.10, 0.15, 0.20],
    },
    "Mean Reversion": {
        "z_threshold": [1.0, 1.5, 2.0, 2.5],
        "lookback":    [10, 20, 30],
    },
    "Combined Best": {
        "w_peer":   [0.2, 0.3, 0.4],
        "w_contra": [0.2, 0.3, 0.4],
        "w_mtf":    [0.1, 0.2, 0.3],
        "w_mr":     [0.1, 0.2, 0.3],
    },
}

# Map strategy name -> (wrapper_fn, needs_df_flag)
_STRATEGY_MAP = {
    "Momentum+Stop":      (momentum_stop_params,      False),
    "Market Contrarian":  (market_contrarian_params,   False),
    "ML Direction":       (ml_direction_params,        True),
    "Peer Lead-Lag":      (peer_leadlag_params,        False),
    "Multi-TF Momentum":  (multi_tf_momentum_params,   False),
    "Mean Reversion":     (mean_reversion_params,      False),
    "Combined Best":      (combined_best_params,       False),
}


def _grid_combos(grid: dict):
    """Yield all parameter dicts from a grid."""
    keys = list(grid.keys())
    for vals in product(*grid.values()):
        yield dict(zip(keys, vals))


# =====================================================================
# GRID SEARCH
# =====================================================================

def run_grid_search(data, df, feature_cols):
    """
    For every strategy, search parameter grid on Train+Val, then evaluate on Test.
    Returns (all_results, best_per_strat, test_results).
    """
    onds = data["onds"]
    onds_close = onds["Close"]
    idx = onds_close.index
    train_end, val_end = make_split(idx)
    train_start = idx[0]
    test_start  = idx[idx > val_end][0] if (idx > val_end).any() else val_end
    test_end    = idx[-1]

    print(f"  Train : {train_start.date()} to {train_end.date()}  "
          f"({(idx <= train_end).sum()} days)")
    n_val = ((idx > train_end) & (idx <= val_end)).sum()
    print(f"  Valid : {idx[idx > train_end][0].date()} to {val_end.date()}  "
          f"({n_val} days)")
    print(f"  Test  : {test_start.date()} to {test_end.date()}  "
          f"({(idx > val_end).sum()} days)")
    print()

    all_rows = []
    best_per_strat = {}
    test_results = []

    for strat_name, grid in GRIDS.items():
        wrapper_fn, needs_df = _STRATEGY_MAP[strat_name]
        combos = list(_grid_combos(grid))
        print(f"  {strat_name}: {len(combos)} combinations ...", end=" ", flush=True)

        best_score = -999
        best_params = None
        best_train_sharpe = None
        best_val_sharpe = None

        for params in combos:
            # Generate signal on FULL data (rolling indicators need history)
            try:
                if needs_df:
                    sig = wrapper_fn(df, feature_cols, params)
                else:
                    sig = wrapper_fn(data, params)
            except Exception:
                continue

            # Backtest on TRAIN period
            r_train = period_backtest(
                onds_close, sig, train_start, train_end, name=strat_name,
            )
            # Backtest on VALIDATION period
            val_start_date = idx[idx > train_end][0]
            r_val = period_backtest(
                onds_close, sig, val_start_date, val_end, name=strat_name,
            )

            if r_train is None or r_val is None:
                continue

            train_sharpe = r_train["sharpe"]
            val_sharpe   = r_val["sharpe"]
            combined     = 0.5 * train_sharpe + 0.5 * val_sharpe

            row = {
                "strategy": strat_name,
                "params":   str(params),
                "train_sharpe": train_sharpe,
                "train_return": r_train["total_return"],
                "val_sharpe":   val_sharpe,
                "val_return":   r_val["total_return"],
                "combined_score": combined,
            }
            all_rows.append(row)

            if train_sharpe > 0 and val_sharpe > 0 and combined > best_score:
                best_score = combined
                best_params = params
                best_train_sharpe = train_sharpe
                best_val_sharpe = val_sharpe

        if best_params is not None:
            best_per_strat[strat_name] = {
                "params": best_params,
                "train_sharpe": best_train_sharpe,
                "val_sharpe": best_val_sharpe,
                "combined_score": best_score,
            }
            print(f"best combined={best_score:+.3f}  params={best_params}")

            # Final TEST evaluation with best params
            try:
                if needs_df:
                    sig_best = wrapper_fn(df, feature_cols, best_params)
                else:
                    sig_best = wrapper_fn(data, best_params)

                # Raw test
                r_test = period_backtest(
                    onds_close, sig_best, test_start, test_end,
                    name=f"{strat_name} (Optimized)",
                )
                # VolAdj test
                sig_va = vol_adjusted_signal(sig_best, onds_close)
                r_test_va = period_backtest(
                    onds_close, sig_va, test_start, test_end,
                    name=f"{strat_name} (Opt+VolAdj)",
                )
                # Buy-and-hold benchmark on test
                bh = period_backtest(
                    onds_close,
                    pd.Series(1.0, index=onds_close.index),
                    test_start, test_end, name="Buy&Hold",
                )

                if r_test:
                    test_results.append({
                        "strategy": strat_name,
                        "params": str(best_params),
                        "test_sharpe": r_test["sharpe"],
                        "test_return": r_test["total_return"],
                        "test_maxdd": r_test["max_drawdown"],
                        "test_hit": r_test["hit_rate"],
                        "test_trades": r_test["n_trades"],
                        "test_va_sharpe": r_test_va["sharpe"] if r_test_va else None,
                        "test_va_return": r_test_va["total_return"] if r_test_va else None,
                        "bh_return": bh["total_return"] if bh else None,
                        "alpha": (r_test["total_return"] - bh["total_return"])
                                 if bh else None,
                        "train_sharpe": best_train_sharpe,
                        "val_sharpe": best_val_sharpe,
                        "_bt": r_test,
                        "_bt_va": r_test_va,
                        "_bh": bh,
                    })
            except Exception as e:
                print(f"    TEST eval failed: {e}")
        else:
            print("no valid params (train/val both > 0)")

    return all_rows, best_per_strat, test_results


# =====================================================================
# VISUALISATION
# =====================================================================

def plot_param_sensitivity(all_rows, save=True):
    """Heatmaps of Sharpe across the two most important params per strategy."""
    df = pd.DataFrame(all_rows)
    strategies = df["strategy"].unique()
    n = len(strategies)
    if n == 0:
        return

    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(6 * ((n + 1) // 2), 10))
    axes = np.array(axes).flatten()

    for idx, strat in enumerate(strategies):
        ax = axes[idx]
        sub = df[df["strategy"] == strat].copy()

        # Parse params string back to dict
        import ast
        sub["_p"] = sub["params"].apply(ast.literal_eval)

        # Pick the first two keys for heatmap axes
        keys = list(GRIDS.get(strat, {}).keys())
        if len(keys) < 2:
            ax.set_title(strat)
            ax.text(0.5, 0.5, "1-D grid", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        k1, k2 = keys[0], keys[1]
        sub["k1"] = sub["_p"].apply(lambda p: str(p.get(k1, "")))
        sub["k2"] = sub["_p"].apply(lambda p: str(p.get(k2, "")))

        # Average combined score across other params
        pivot = sub.pivot_table(
            values="combined_score", index="k2", columns="k1", aggfunc="mean",
        )
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", origin="lower")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=7, rotation=45)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=7)
        ax.set_xlabel(k1, fontsize=8)
        ax.set_ylabel(k2, fontsize=8)
        ax.set_title(strat, fontsize=9, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for j in range(idx + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Parameter Sensitivity — Combined Sharpe (Train+Val)", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save:
        path = FIGURES_DIR / "param_sensitivity.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


def plot_test_equity(test_results, save=True):
    """Overlay equity curves of optimised strategies on the test period."""
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(test_results), 1)))

    for i, tr in enumerate(test_results):
        bt = tr.get("_bt")
        if bt and "cum_return" in bt:
            ax.plot(bt["cum_return"], label=f"{tr['strategy']} (S={bt['sharpe']:.2f})",
                    color=colors[i], linewidth=1.5)
        bt_va = tr.get("_bt_va")
        if bt_va and "cum_return" in bt_va:
            ax.plot(bt_va["cum_return"],
                    label=f"{tr['strategy']} VA (S={bt_va['sharpe']:.2f})",
                    color=colors[i], linewidth=1, linestyle="--", alpha=0.7)

    # Buy & hold
    bh = test_results[0].get("_bh") if test_results else None
    if bh and "cum_return" in bh:
        ax.plot(bh["cum_return"], label="Buy & Hold", color="gray",
                linewidth=1.5, linestyle=":")

    ax.set_title("Optimised Strategy Equity Curves — Test Period (30%)")
    ax.set_ylabel("Cumulative Return")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        path = FIGURES_DIR / "optimized_test_equity.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


# =====================================================================
# MAIN
# =====================================================================

def run_param_optimization():
    print("=" * 70)
    print("  PARAMETER OPTIMISATION — 60/10/30 Split")
    print("=" * 70)

    # ── Load ──────────────────────────────────────────────────────
    print("\n[1/5] Loading data...")
    data = load_all_data()

    print("\n[2/5] Building features (for ML strategy)...")
    df = build_advanced_features(data)

    # Feature selection for ML wrapper
    leakage = {"ret_open_close", "ret_high_low", "ret_close_high", "ret_close_low",
               "gap", "volume_change", "volume_ma_ratio", "dollar_volume",
               "price_range_pct", "ret_1d", "dow", "is_monday", "is_friday"}
    all_feat = [c for c in df.columns
                if not c.startswith("target_") and c not in leakage]
    feature_cols = select_features(df, "target_direction", n_top=20,
                                   allowed_cols=all_feat)
    print(f"  ML feature set: {feature_cols[:5]}... ({len(feature_cols)} total)")

    # ── Grid search ───────────────────────────────────────────────
    print("\n[3/5] Running grid search...")
    all_rows, best_per_strat, test_results = run_grid_search(data, df, feature_cols)

    # ── Save CSVs ─────────────────────────────────────────────────
    print("\n[4/5] Saving results...")
    grid_df = pd.DataFrame(all_rows)
    grid_df.to_csv(REPORTS_DIR / "param_optimization_results.csv", index=False)
    print(f"  Saved: {REPORTS_DIR / 'param_optimization_results.csv'}  "
          f"({len(grid_df)} rows)")

    test_df = pd.DataFrame([
        {k: v for k, v in r.items() if not k.startswith("_")}
        for r in test_results
    ])
    test_df.to_csv(REPORTS_DIR / "optimized_test_results.csv", index=False)
    print(f"  Saved: {REPORTS_DIR / 'optimized_test_results.csv'}  "
          f"({len(test_df)} rows)")

    # ── Figures ───────────────────────────────────────────────────
    print("\n[5/5] Plotting...")
    plot_param_sensitivity(all_rows)
    if test_results:
        plot_test_equity(test_results)

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  OPTIMISED TEST PERFORMANCE")
    print("=" * 70)
    header = (f"  {'Strategy':<22s} {'Test Sharpe':>11s} {'Test Ret':>9s} "
              f"{'MaxDD':>7s} {'Alpha':>7s} {'Trn S':>7s} {'Val S':>7s}")
    print(header)
    print("  " + "-" * len(header.strip()))
    for tr in sorted(test_results, key=lambda x: x["test_sharpe"], reverse=True):
        alpha_str = f"{tr['alpha']:+.2%}" if tr["alpha"] is not None else "N/A"
        print(f"  {tr['strategy']:<22s} {tr['test_sharpe']:>+11.3f} "
              f"{tr['test_return']:>+9.2%} {tr['test_maxdd']:>7.2%} "
              f"{alpha_str:>7s} {tr['train_sharpe']:>+7.3f} "
              f"{tr['val_sharpe']:>+7.3f}")

    print("\n  Best params per strategy:")
    for name, info in best_per_strat.items():
        print(f"    {name}: {info['params']}")

    return all_rows, best_per_strat, test_results


if __name__ == "__main__":
    run_param_optimization()
