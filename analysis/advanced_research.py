"""
Advanced Quantitative Research for ONDS
========================================
Comprehensive exploration of every possible prediction angle:
  1. Multiple prediction targets (direction, magnitude, volatility, range)
  2. Advanced feature engineering (lags, interactions, selections)
  3. Advanced models (XGBoost, LightGBM, feature-selected lean models)
  4. Regime-conditional strategies
  5. Peer lead-lag exploitation (JOBY, RCAT, AVAV)
  6. Pair trading
  7. Volatility-targeted position sizing
  8. Mean-reversion and momentum variants
"""
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               RandomForestRegressor, GradientBoostingRegressor)
from sklearn.linear_model import LogisticRegression, Lasso, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
import sys

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_RAW, DATA_PROC, DATA_FEAT, FIGURES_DIR, REPORTS_DIR, BACKTEST
from backtests.engine import backtest, print_results, compare_strategies


# ═══════════════════════════════════════════════════════════════════
# SECTION 1: DATA LOADING
# ═══════════════════════════════════════════════════════════════════

def load_all_data():
    """Load all available data sources for advanced analysis."""
    from collectors.prices import load_prices, load_all_closes

    data = {}
    data["onds"] = load_prices("ONDS")
    data["closes"] = load_all_closes()

    # Load peer prices
    for ticker in ["RCAT", "JOBY", "AVAV", "KTOS", "LMT", "RTX"]:
        try:
            data[ticker.lower()] = load_prices(ticker)
        except:
            pass

    # Cross-asset
    for ticker in ["GLD", "SLV", "BTC_USD", "SPY", "QQQ", "TLT", "UUP"]:
        try:
            data[ticker.lower()] = load_prices(ticker)
        except:
            pass

    # VIX
    try:
        data["vix"] = load_prices("VIX")
    except:
        pass

    # Short volume
    try:
        sv = pd.read_csv(DATA_RAW / "chartexchange_onds_short_volume.csv")
        sv["date"] = pd.to_datetime(sv["date"], format="%Y%m%d")
        sv = sv.set_index("date").sort_index()
        sv["short_ratio"] = sv["total_short"] / sv["total_reported"]
        data["short_volume"] = sv
    except:
        pass

    # DIX/GEX
    try:
        dix = pd.read_csv(DATA_RAW / "squeezemetrics_dix_gex.csv", index_col=0, parse_dates=True)
        data["dix"] = dix
    except:
        pass

    # Regime data
    try:
        data["regime_hmm"] = pd.read_csv(DATA_PROC / "regimes_hmm.csv", index_col=0, parse_dates=True)
    except:
        pass

    print(f"  Loaded {len(data)} data sources")
    return data


# ═══════════════════════════════════════════════════════════════════
# SECTION 2: ADVANCED FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════

def build_advanced_features(data: dict) -> pd.DataFrame:
    """
    Build comprehensive feature matrix with:
    - OHLCV features (not just Close)
    - Lagged features (1-5 days)
    - Rolling statistics
    - Cross-asset signals
    - Peer lead-lag
    - Short volume
    - Interaction terms
    """
    onds = data["onds"]
    df = pd.DataFrame(index=onds.index)

    # ── OHLCV Features ──────────────────────────────────────────
    df["ret_1d"] = onds["Close"].pct_change()
    df["ret_open_close"] = (onds["Close"] - onds["Open"]) / onds["Open"]  # intraday return
    df["ret_high_low"] = (onds["High"] - onds["Low"]) / onds["Low"]  # daily range
    df["ret_close_high"] = (onds["High"] - onds["Close"]) / onds["Close"]  # upper shadow
    df["ret_close_low"] = (onds["Close"] - onds["Low"]) / onds["Low"]  # lower shadow
    df["gap"] = (onds["Open"] - onds["Close"].shift(1)) / onds["Close"].shift(1)  # overnight gap
    df["volume_change"] = onds["Volume"].pct_change()
    df["volume_ma_ratio"] = onds["Volume"] / onds["Volume"].rolling(20).mean()
    df["dollar_volume"] = onds["Close"] * onds["Volume"]
    df["price_range_pct"] = (onds["High"] - onds["Low"]) / onds["Close"]

    # ── Lagged Returns ──────────────────────────────────────────
    for lag in [1, 2, 3, 5]:
        df[f"ret_lag{lag}"] = df["ret_1d"].shift(lag)
        df[f"gap_lag{lag}"] = df["gap"].shift(lag)
        df[f"range_lag{lag}"] = df["ret_high_low"].shift(lag)

    # ── Rolling Statistics ──────────────────────────────────────
    for window in [5, 10, 20]:
        df[f"ret_mean_{window}d"] = df["ret_1d"].rolling(window).mean()
        df[f"ret_std_{window}d"] = df["ret_1d"].rolling(window).std()
        df[f"ret_skew_{window}d"] = df["ret_1d"].rolling(window).skew()
        df[f"vol_mean_{window}d"] = onds["Volume"].rolling(window).mean()
        df[f"range_mean_{window}d"] = df["ret_high_low"].rolling(window).mean()
        df[f"high_pct_{window}d"] = (onds["Close"] / onds["Close"].rolling(window).max()) - 1
        df[f"low_pct_{window}d"] = (onds["Close"] / onds["Close"].rolling(window).min()) - 1

    # ── RSI ─────────────────────────────────────────────────────
    delta = onds["Close"].diff()
    gain = delta.clip(lower=0).ewm(span=14).mean()
    loss = (-delta).clip(lower=0).ewm(span=14).mean()
    df["rsi_14"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    # ── MACD ────────────────────────────────────────────────────
    ema12 = onds["Close"].ewm(span=12).mean()
    ema26 = onds["Close"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # ── Bollinger ───────────────────────────────────────────────
    sma20 = onds["Close"].rolling(20).mean()
    std20 = onds["Close"].rolling(20).std()
    df["bb_pct"] = (onds["Close"] - sma20) / (2 * std20)

    # ── Peer Lead-Lag Features ──────────────────────────────────
    # IMPORTANT: reindex closes to onds.index FIRST, then compute pct_change
    # This avoids NaN Monday returns caused by weekend gaps in BTC-USD rows
    closes_raw = data.get("closes", pd.DataFrame())
    closes = closes_raw.reindex(onds.index)

    for peer in ["RCAT", "JOBY", "AVAV", "KTOS", "LMT", "RTX"]:
        if peer in closes.columns:
            peer_ret = closes[peer].pct_change()
            for lag in [1, 2, 3, 5]:
                df[f"{peer}_ret_lag{lag}"] = peer_ret.shift(lag)
            df[f"{peer}_ret_5d"] = closes[peer].pct_change(5).shift(1)

    # Peer average and dispersion
    peer_cols = [c for c in closes.columns if c not in ["ONDS"]]
    if peer_cols:
        peer_rets = closes[peer_cols].pct_change()
        df["peer_avg_ret_lag1"] = peer_rets.mean(axis=1).shift(1)
        df["peer_std_ret_lag1"] = peer_rets.std(axis=1).shift(1)

    # ── Cross-Asset Features ────────────────────────────────────
    cross_map = {"SPY": "sp500", "QQQ": "nasdaq", "TLT": "bonds",
                 "GLD": "gold", "SLV": "silver", "UUP": "usd"}
    for ticker, name in cross_map.items():
        if ticker in closes.columns:
            ret = closes[ticker].pct_change()
            df[f"{name}_ret_lag1"] = ret.shift(1)
            df[f"{name}_ret_5d"] = closes[ticker].pct_change(5).shift(1)

    # VIX level and change
    vix_col = "^VIX" if "^VIX" in closes.columns else ("VIX" if "VIX" in closes.columns else None)
    if vix_col:
        df["vix_level"] = closes[vix_col]
        df["vix_change"] = closes[vix_col].pct_change().shift(1)
        df["vix_5d_change"] = closes[vix_col].pct_change(5).shift(1)

    # ── DIX/GEX ─────────────────────────────────────────────────
    if "dix" in data:
        dix_df = data["dix"]
        df["dix"] = dix_df["dix"]
        df["gex"] = dix_df["gex"]
        df["dix_z"] = (dix_df["dix"] - dix_df["dix"].rolling(60).mean()) / dix_df["dix"].rolling(60).std()
        df["gex_z"] = (dix_df["gex"] - dix_df["gex"].rolling(60).mean()) / dix_df["gex"].rolling(60).std()

    # ── Short Volume ────────────────────────────────────────────
    if "short_volume" in data:
        sv = data["short_volume"]
        df["short_ratio"] = sv["short_ratio"]
        df["short_ratio_z"] = (sv["short_ratio"] - sv["short_ratio"].rolling(20).mean()) / sv["short_ratio"].rolling(20).std()
        df["finra_volume"] = sv["total_reported"]

    # ── Regime ──────────────────────────────────────────────────
    if "regime_hmm" in data:
        regime = data["regime_hmm"]
        if "regime" in regime.columns:
            df["regime"] = regime["regime"]

    # ── Day-of-Week ─────────────────────────────────────────────
    df["dow"] = df.index.dayofweek
    df["is_monday"] = (df["dow"] == 0).astype(int)
    df["is_friday"] = (df["dow"] == 4).astype(int)

    # ── Targets ─────────────────────────────────────────────────
    df["target_direction"] = (df["ret_1d"].shift(-1) > 0).astype(int)  # next-day close direction
    df["target_return"] = df["ret_1d"].shift(-1)  # next-day return magnitude
    df["target_abs_return"] = df["ret_1d"].shift(-1).abs()  # next-day volatility
    df["target_range"] = df["ret_high_low"].shift(-1)  # next-day high-low range

    # Next-day open-to-close
    df["target_intraday"] = df["ret_open_close"].shift(-1)
    df["target_intraday_dir"] = (df["target_intraday"] > 0).astype(int)

    # Multi-class: big-up / small / big-down
    next_ret = df["ret_1d"].shift(-1)
    threshold = next_ret.std() * 0.5
    df["target_3class"] = 1  # neutral
    df.loc[next_ret > threshold, "target_3class"] = 2  # up
    df.loc[next_ret < -threshold, "target_3class"] = 0  # down

    # Gap prediction: will tomorrow open higher or lower?
    df["target_gap_dir"] = (df["gap"].shift(-1) > 0).astype(int)

    print(f"  Advanced feature matrix: {df.shape[0]} rows × {df.shape[1]} columns")
    target_cols = [c for c in df.columns if c.startswith("target_")]
    feature_cols = [c for c in df.columns if not c.startswith("target_")]
    print(f"  Features: {len(feature_cols)}, Targets: {len(target_cols)}")

    return df


# ═══════════════════════════════════════════════════════════════════
# SECTION 3: FEATURE SELECTION
# ═══════════════════════════════════════════════════════════════════

def select_features(df: pd.DataFrame, target: str, n_top: int = 20,
                     allowed_cols: list = None) -> list:
    """Select top features using mutual information."""
    if allowed_cols is not None:
        feature_cols = [c for c in allowed_cols if c in df.columns]
    else:
        feature_cols = [c for c in df.columns if not c.startswith("target_") and c != "dow"]
    clean = df[feature_cols + [target]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 50:
        return feature_cols[:n_top]

    X = clean[feature_cols].values
    y = clean[target].values

    if target in ["target_direction", "target_intraday_dir", "target_gap_dir", "target_3class"]:
        mi = mutual_info_classif(X, y, random_state=42, n_neighbors=5)
    else:
        mi = mutual_info_regression(X, y, random_state=42, n_neighbors=5)

    mi_series = pd.Series(mi, index=feature_cols).sort_values(ascending=False)
    top = mi_series.head(n_top).index.tolist()
    return top


# ═══════════════════════════════════════════════════════════════════
# SECTION 4: MODEL TRAINING & EVALUATION
# ═══════════════════════════════════════════════════════════════════

def evaluate_classification(df, feature_cols, target_col, n_splits=5, label="Model"):
    """Walk-forward classification evaluation."""
    clean = df[feature_cols + [target_col]].replace([np.inf, -np.inf], np.nan).dropna()
    X = clean[feature_cols]
    y = clean[target_col]

    if len(X) < 80:
        return None

    tscv = TimeSeriesSplit(n_splits=n_splits)
    scaler = StandardScaler()

    models = {
        "RF": RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_leaf=10, random_state=42),
        "GB": GradientBoostingClassifier(n_estimators=150, max_depth=3, min_samples_leaf=10,
                                          learning_rate=0.05, random_state=42),
        "LR": LogisticRegression(C=0.1, max_iter=1000, random_state=42),
    }

    # Try XGBoost if available
    try:
        from xgboost import XGBClassifier
        models["XGB"] = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                       min_child_weight=10, reg_alpha=0.1, reg_lambda=1.0,
                                       random_state=42, verbosity=0)
    except ImportError:
        pass

    results = {}
    for name, model in models.items():
        scores = []
        probs_all = []
        y_all = []
        for train_idx, test_idx in tscv.split(X):
            X_train = scaler.fit_transform(X.iloc[train_idx])
            X_test = scaler.transform(X.iloc[test_idx])
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            scores.append(accuracy_score(y_test, pred))
            if hasattr(model, 'predict_proba'):
                probs_all.extend(model.predict_proba(X_test)[:, -1])
                y_all.extend(y_test.values)

        results[name] = {
            "accuracy": np.mean(scores),
            "std": np.std(scores),
            "scores": scores,
        }

    best_name = max(results, key=lambda k: results[k]["accuracy"])
    best = results[best_name]
    return {
        "label": label,
        "target": target_col,
        "n_features": len(feature_cols),
        "n_samples": len(X),
        "best_model": best_name,
        "best_accuracy": best["accuracy"],
        "best_std": best["std"],
        "all_results": results,
        "feature_cols": feature_cols,
    }


def evaluate_regression(df, feature_cols, target_col, n_splits=5, label="Model"):
    """Walk-forward regression evaluation."""
    clean = df[feature_cols + [target_col]].replace([np.inf, -np.inf], np.nan).dropna()
    X = clean[feature_cols]
    y = clean[target_col]

    if len(X) < 80:
        return None

    tscv = TimeSeriesSplit(n_splits=n_splits)
    scaler = StandardScaler()

    models = {
        "RF_reg": RandomForestRegressor(n_estimators=200, max_depth=4, min_samples_leaf=10, random_state=42),
        "GB_reg": GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.001, max_iter=5000),
    }

    try:
        from xgboost import XGBRegressor
        models["XGB_reg"] = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                          min_child_weight=10, random_state=42, verbosity=0)
    except ImportError:
        pass

    results = {}
    for name, model in models.items():
        r2_scores = []
        directional_acc = []
        for train_idx, test_idx in tscv.split(X):
            X_train = scaler.fit_transform(X.iloc[train_idx])
            X_test = scaler.transform(X.iloc[test_idx])
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            r2_scores.append(r2_score(y_test, pred))
            # Directional accuracy: does predicted sign match actual sign?
            dir_acc = ((pred > 0) == (y_test > 0)).mean()
            directional_acc.append(dir_acc)

        results[name] = {
            "r2": np.mean(r2_scores),
            "r2_std": np.std(r2_scores),
            "dir_acc": np.mean(directional_acc),
            "dir_acc_std": np.std(directional_acc),
        }

    best_name = max(results, key=lambda k: results[k]["dir_acc"])
    best = results[best_name]
    return {
        "label": label,
        "target": target_col,
        "n_features": len(feature_cols),
        "n_samples": len(X),
        "best_model": best_name,
        "best_r2": best["r2"],
        "best_dir_acc": best["dir_acc"],
        "best_dir_acc_std": best["dir_acc_std"],
        "all_results": results,
        "feature_cols": feature_cols,
    }


# ═══════════════════════════════════════════════════════════════════
# SECTION 5: STRATEGY GENERATORS
# ═══════════════════════════════════════════════════════════════════

def strategy_peer_leadlag(data: dict) -> pd.Series:
    """
    Strategy: exploit JOBY lag-1 (p=0.007) and RCAT lag-2 (p=0.045).
    JOBY lag-1 has NEGATIVE correlation → contrarian.
    """
    onds = data["onds"]
    closes = data["closes"].reindex(onds.index)
    sig = pd.Series(0.0, index=onds.index)

    if "JOBY" in closes.columns:
        joby_ret = closes["JOBY"].pct_change()
        joby_lag1 = joby_ret.shift(1)
        joby_z = (joby_lag1 - joby_lag1.rolling(20).mean()) / joby_lag1.rolling(20).std()
        joby_sig = pd.Series(np.where(joby_z < -0.5, +0.5, np.where(joby_z > 0.5, -0.5, 0)),
                             index=onds.index)
        sig += joby_sig.fillna(0)

    if "RCAT" in closes.columns:
        rcat_ret = closes["RCAT"].pct_change()
        rcat_lag2 = rcat_ret.shift(2)
        rcat_z = (rcat_lag2 - rcat_lag2.rolling(20).mean()) / rcat_lag2.rolling(20).std()
        rcat_sig = pd.Series(np.where(rcat_z < -0.5, +0.3, np.where(rcat_z > 0.5, -0.3, 0)),
                             index=onds.index)
        sig += rcat_sig.fillna(0)

    return sig.clip(-1, 1)


def strategy_market_contrarian(data: dict) -> pd.Series:
    """
    Strategy: NASDAQ/SP500 same-day has negative correlation with ONDS next-day.
    If NASDAQ is up big today → short ONDS tomorrow (and vice versa).
    """
    onds = data["onds"]
    closes = data["closes"].reindex(onds.index)
    sig = pd.Series(0.0, index=onds.index)

    if "QQQ" in closes.columns:
        nasdaq_ret = closes["QQQ"].pct_change()
        nasdaq_z = (nasdaq_ret - nasdaq_ret.rolling(20).mean()) / nasdaq_ret.rolling(20).std()
        nasdaq_sig = pd.Series(np.where(nasdaq_z > 0.75, -0.5, np.where(nasdaq_z < -0.75, +0.5, 0)),
                               index=onds.index)
        sig += nasdaq_sig.fillna(0)

    if "SPY" in closes.columns:
        sp_ret = closes["SPY"].pct_change()
        sp_z = (sp_ret - sp_ret.rolling(20).mean()) / sp_ret.rolling(20).std()
        sp_sig = pd.Series(np.where(sp_z > 0.75, -0.3, np.where(sp_z < -0.75, +0.3, 0)),
                           index=onds.index)
        sig += sp_sig.fillna(0)

    return sig.clip(-1, 1)


def strategy_regime_conditional(data: dict) -> pd.Series:
    """
    Strategy: only trade during favorable regimes.
    Bearish regime (86% of time) → stay flat or short.
    Bullish regime (6%) → go long aggressively.
    Neutral → small long.
    """
    onds = data["onds"]
    sig = pd.Series(0.0, index=onds.index)

    if "regime_hmm" in data:
        regime = data["regime_hmm"].reindex(onds.index)
        if "regime" in regime.columns:
            # Map regimes by their mean return
            regime_stats = {}
            onds_ret = onds["Close"].pct_change()
            for r_val in regime["regime"].dropna().unique():
                mask = regime["regime"] == r_val
                common_idx = mask.index.intersection(onds_ret.index)
                mean_ret = onds_ret.loc[common_idx][mask.loc[common_idx]].mean()
                regime_stats[r_val] = mean_ret

            # Sort regimes by mean return
            sorted_regimes = sorted(regime_stats.items(), key=lambda x: x[1])
            bearish = sorted_regimes[0][0]
            bullish = sorted_regimes[-1][0]

            for date in sig.index:
                if date in regime.index and not pd.isna(regime.loc[date, "regime"]):
                    r_val = regime.loc[date, "regime"]
                    if r_val == bullish:
                        sig.loc[date] = 1.0
                    elif r_val == bearish:
                        sig.loc[date] = -0.5
                    else:
                        sig.loc[date] = 0.3

    return sig.clip(-1, 1)


def strategy_volatility_breakout(data: dict) -> pd.Series:
    """
    Strategy: trade breakouts from low-volatility periods.
    After compressed volatility, go long on upward breakout.
    Uses ATR contraction then expansion pattern.
    """
    onds = data["onds"]
    sig = pd.Series(0.0, index=onds.index)

    ret = onds["Close"].pct_change()
    vol_5d = ret.rolling(5).std()
    vol_20d = ret.rolling(20).std()
    vol_ratio = vol_5d / vol_20d

    # Low vol ratio → compression → breakout expected
    for i in range(21, len(sig)):
        if vol_ratio.iloc[i] < 0.6:  # volatility compressed
            # Direction from momentum
            mom = ret.iloc[i-4:i+1].sum()
            if mom > 0:
                sig.iloc[i] = 1.0
            else:
                sig.iloc[i] = -1.0
        elif vol_ratio.iloc[i] > 1.5:  # high vol → mean revert
            if ret.iloc[i] > 0:
                sig.iloc[i] = -0.5  # fade the move
            else:
                sig.iloc[i] = 0.5

    return sig.clip(-1, 1)


def strategy_dix_enhanced(data: dict) -> pd.Series:
    """
    Enhanced DIX strategy: use rolling quintile of DIX + GEX interaction.
    DIX Q5 (high) → +51.8% mean 20d return (from our analysis).
    """
    onds = data["onds"]
    sig = pd.Series(0.0, index=onds.index)

    if "dix" not in data:
        return sig

    dix_df = data["dix"]
    dix = dix_df["dix"].reindex(onds.index)
    gex = dix_df["gex"].reindex(onds.index) if "gex" in dix_df.columns else None

    # Rolling percentile of DIX
    dix_pctl = dix.rolling(120, min_periods=30).rank(pct=True)

    for i in range(30, len(sig)):
        d = sig.index[i]
        if pd.isna(dix_pctl.iloc[i]):
            continue
        pctl = dix_pctl.iloc[i]
        if pctl > 0.8:  # top quintile → strong long
            sig.iloc[i] = 1.0
        elif pctl > 0.6:
            sig.iloc[i] = 0.5
        elif pctl < 0.2:  # bottom quintile → short
            sig.iloc[i] = -0.5
        elif pctl < 0.4:
            sig.iloc[i] = -0.25

        # GEX amplifier
        if gex is not None and not pd.isna(gex.iloc[i]):
            gex_pctl = gex.rolling(120, min_periods=30).rank(pct=True).iloc[i]
            if not pd.isna(gex_pctl) and gex_pctl < 0.2:
                sig.iloc[i] *= 1.5  # low GEX amplifies

    return sig.clip(-1, 1)


def strategy_mean_reversion(data: dict) -> pd.Series:
    """
    Strategy: ONDS tends to mean-revert after extreme moves.
    After a big up day, expect pullback. After big down, expect bounce.
    """
    onds = data["onds"]
    ret = onds["Close"].pct_change()
    sig = pd.Series(0.0, index=onds.index)

    # Z-score of daily return relative to recent history
    ret_z = (ret - ret.rolling(20).mean()) / ret.rolling(20).std()

    for i in range(20, len(sig)):
        z = ret_z.iloc[i]
        if z > 2:  # extreme up → short (expect reversion)
            sig.iloc[i] = -1.0
        elif z > 1:
            sig.iloc[i] = -0.5
        elif z < -2:  # extreme down → long (expect bounce)
            sig.iloc[i] = 1.0
        elif z < -1:
            sig.iloc[i] = 0.5

    return sig


def strategy_momentum_with_stop(data: dict) -> pd.Series:
    """
    Strategy: trend-following with dynamic stops.
    Go long when price > SMA10 and SMA10 > SMA50.
    Exit when price drops below SMA10 or 10% trailing stop hit.
    """
    onds = data["onds"]
    close = onds["Close"]
    sma10 = close.rolling(10).mean()
    sma50 = close.rolling(50).mean()
    sig = pd.Series(0.0, index=onds.index)

    position = 0
    entry_price = 0
    peak_price = 0

    for i in range(50, len(sig)):
        price = close.iloc[i]

        if position == 0:
            # Entry: price > SMA10 > SMA50 (uptrend)
            if price > sma10.iloc[i] and sma10.iloc[i] > sma50.iloc[i]:
                position = 1
                entry_price = price
                peak_price = price
        elif position == 1:
            peak_price = max(peak_price, price)
            # Exit conditions
            if price < sma10.iloc[i]:  # trend break
                position = 0
            elif price < peak_price * 0.90:  # 10% trailing stop
                position = 0
            elif price < entry_price * 0.85:  # 15% max loss stop
                position = 0

        sig.iloc[i] = position

    return sig


def strategy_gap_fade(data: dict) -> pd.Series:
    """
    Strategy: fade overnight gaps.
    If ONDS gaps up big at open → short (expect gap fill).
    If gaps down big → long.
    """
    onds = data["onds"]
    gap = (onds["Open"] - onds["Close"].shift(1)) / onds["Close"].shift(1)
    sig = pd.Series(0.0, index=onds.index)

    gap_std = gap.rolling(20).std()
    gap_z = (gap - gap.rolling(20).mean()) / gap_std

    for i in range(20, len(sig)):
        z = gap_z.iloc[i]
        if z > 1.5:  # big gap up → fade
            sig.iloc[i] = -1.0
        elif z < -1.5:  # big gap down → fade
            sig.iloc[i] = 1.0

    return sig


def strategy_volume_spike(data: dict) -> pd.Series:
    """
    Strategy: unusual volume + price action.
    High volume + up move → momentum continuation.
    High volume + down move → potential reversal (capitulation).
    """
    onds = data["onds"]
    ret = onds["Close"].pct_change()
    vol_ratio = onds["Volume"] / onds["Volume"].rolling(20).mean()
    sig = pd.Series(0.0, index=onds.index)

    for i in range(20, len(sig)):
        vr = vol_ratio.iloc[i]
        r = ret.iloc[i]

        if vr > 2.0:  # volume spike (2x average)
            if r > 0.05:  # up with volume → momentum
                sig.iloc[i] = 1.0
            elif r < -0.05:  # down with volume → capitulation → bounce
                sig.iloc[i] = 0.5
            elif r < -0.10:  # severe down with volume
                sig.iloc[i] = 1.0  # strong bounce expected

    return sig


def strategy_rsi_divergence(data: dict) -> pd.Series:
    """
    Strategy: RSI divergence — price makes new low but RSI doesn't (bullish divergence),
    or price makes new high but RSI doesn't (bearish divergence).
    """
    onds = data["onds"]
    close = onds["Close"]
    sig = pd.Series(0.0, index=onds.index)

    # Compute RSI
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(span=14).mean()
    loss = (-delta).clip(lower=0).ewm(span=14).mean()
    rsi = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    lookback = 10
    for i in range(lookback + 14, len(sig)):
        window_price = close.iloc[i-lookback:i+1]
        window_rsi = rsi.iloc[i-lookback:i+1]

        # Bullish divergence: price at local low but RSI higher than previous low
        if (close.iloc[i] <= window_price.min() * 1.02 and
            rsi.iloc[i] > window_rsi.min() + 5 and
            rsi.iloc[i] < 40):
            sig.iloc[i] = 1.0

        # Bearish divergence: price at local high but RSI lower
        elif (close.iloc[i] >= window_price.max() * 0.98 and
              rsi.iloc[i] < window_rsi.max() - 5 and
              rsi.iloc[i] > 60):
            sig.iloc[i] = -1.0

    return sig


def strategy_multi_timeframe_momentum(data: dict) -> pd.Series:
    """
    Strategy: align momentum across multiple timeframes.
    Only go long when 5d, 10d, and 20d momentum ALL agree.
    """
    onds = data["onds"]
    close = onds["Close"]
    sig = pd.Series(0.0, index=onds.index)

    mom5 = close.pct_change(5)
    mom10 = close.pct_change(10)
    mom20 = close.pct_change(20)

    for i in range(20, len(sig)):
        m5, m10, m20 = mom5.iloc[i], mom10.iloc[i], mom20.iloc[i]
        if pd.isna(m5) or pd.isna(m10) or pd.isna(m20):
            continue

        # All timeframes bullish
        if m5 > 0 and m10 > 0 and m20 > 0:
            strength = min(1.0, (m5 + m10 + m20) / 0.15)
            sig.iloc[i] = strength

        # All timeframes bearish
        elif m5 < 0 and m10 < 0 and m20 < 0:
            strength = min(1.0, abs(m5 + m10 + m20) / 0.15)
            sig.iloc[i] = -strength

    return sig.clip(-1, 1)


def strategy_bollinger_mean_reversion(data: dict) -> pd.Series:
    """
    Strategy: Bollinger Band mean reversion.
    Buy when price touches lower band, sell when touches upper band.
    Confirm with volume.
    """
    onds = data["onds"]
    close = onds["Close"]
    volume = onds["Volume"]
    sig = pd.Series(0.0, index=onds.index)

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    vol_avg = volume.rolling(20).mean()

    for i in range(20, len(sig)):
        bb_pct = (close.iloc[i] - lower.iloc[i]) / (upper.iloc[i] - lower.iloc[i])
        vol_ratio = volume.iloc[i] / vol_avg.iloc[i] if vol_avg.iloc[i] > 0 else 1

        if bb_pct < 0.05:  # near lower band
            sig.iloc[i] = min(1.0, 0.5 + 0.5 * vol_ratio / 2)  # higher vol = stronger signal
        elif bb_pct > 0.95:  # near upper band
            sig.iloc[i] = -min(1.0, 0.5 + 0.5 * vol_ratio / 2)
        elif bb_pct < 0.2:
            sig.iloc[i] = 0.3
        elif bb_pct > 0.8:
            sig.iloc[i] = -0.3

    return sig.clip(-1, 1)


def strategy_adaptive_ensemble(data: dict, df: pd.DataFrame) -> pd.Series:
    """
    Adaptive ensemble: weight individual signals by their recent performance.
    Re-evaluate signal weights every 20 days based on trailing 40-day Sharpe.
    """
    onds = data["onds"]
    onds_ret = onds["Close"].pct_change()
    sig = pd.Series(0.0, index=onds.index)

    # Generate all base signals
    base_signals = {
        "regime": strategy_regime_conditional(data),
        "dix": strategy_dix_enhanced(data),
        "momentum": strategy_momentum_with_stop(data),
        "mean_rev": strategy_mean_reversion(data),
        "gap_fade": strategy_gap_fade(data),
        "bb_revert": strategy_bollinger_mean_reversion(data),
        "mtf_mom": strategy_multi_timeframe_momentum(data),
    }

    eval_window = 40
    rebalance_freq = 20

    for i in range(eval_window + 1, len(sig)):
        if (i - eval_window - 1) % rebalance_freq != 0 and i > eval_window + 1:
            # Use previous weights between rebalance dates
            if not pd.isna(sig.iloc[i-1]):
                weighted_sig = 0.0
                for name, s in base_signals.items():
                    if not pd.isna(s.iloc[i]):
                        weighted_sig += s.iloc[i] * weights.get(name, 1.0 / len(base_signals))
                sig.iloc[i] = weighted_sig
                continue

        # Evaluate each signal's recent performance
        weights = {}
        total_weight = 0
        for name, s in base_signals.items():
            # Compute trailing Sharpe of this signal
            trail_pos = s.iloc[i-eval_window:i].shift(1)
            trail_ret = onds_ret.iloc[i-eval_window:i]
            strat_ret = trail_pos * trail_ret
            if strat_ret.std() > 0:
                sharpe = strat_ret.mean() / strat_ret.std()
            else:
                sharpe = 0
            # Convert Sharpe to weight (exponential scaling, positive bias)
            w = max(0, np.exp(sharpe * 2) - 0.5)
            weights[name] = w
            total_weight += w

        # Normalize weights
        if total_weight > 0:
            for name in weights:
                weights[name] /= total_weight
        else:
            for name in weights:
                weights[name] = 1.0 / len(weights)

        # Apply weighted combination
        weighted_sig = 0.0
        for name, s in base_signals.items():
            if not pd.isna(s.iloc[i]):
                weighted_sig += s.iloc[i] * weights[name]
        sig.iloc[i] = weighted_sig

    return sig.clip(-1, 1)


def strategy_ml_signal(df: pd.DataFrame, feature_cols: list, target_col: str) -> pd.Series:
    """Generate walk-forward ML signal from the best model."""
    clean = df[feature_cols + [target_col]].replace([np.inf, -np.inf], np.nan).dropna()
    X = clean[feature_cols]
    y = clean[target_col]

    scaler = StandardScaler()
    signal = pd.Series(0.0, index=clean.index)
    min_train = 60

    for i in range(min_train, len(X)):
        X_train = scaler.fit_transform(X.iloc[:i])
        X_test = scaler.transform(X.iloc[i:i+1])
        y_train = y.iloc[:i]

        rf = RandomForestClassifier(n_estimators=150, max_depth=4, min_samples_leaf=10, random_state=42)
        rf.fit(X_train, y_train)
        prob = rf.predict_proba(X_test)[0]

        if len(prob) == 2:
            p_up = prob[1]
            if p_up > 0.58:
                signal.iloc[i] = 1.0
            elif p_up < 0.42:
                signal.iloc[i] = -1.0

    return signal


def strategy_regression_signal(df: pd.DataFrame, feature_cols: list) -> pd.Series:
    """Walk-forward regression signal: predict return magnitude, trade on size."""
    target_col = "target_return"
    clean = df[feature_cols + [target_col]].replace([np.inf, -np.inf], np.nan).dropna()
    X = clean[feature_cols]
    y = clean[target_col]

    scaler = StandardScaler()
    signal = pd.Series(0.0, index=clean.index)
    min_train = 60

    for i in range(min_train, len(X)):
        X_train = scaler.fit_transform(X.iloc[:i])
        X_test = scaler.transform(X.iloc[i:i+1])
        y_train = y.iloc[:i]

        model = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)[0]

        # Size position by predicted magnitude
        y_std = y_train.std()
        if pred > y_std * 0.3:
            signal.iloc[i] = min(pred / y_std, 1.0)
        elif pred < -y_std * 0.3:
            signal.iloc[i] = max(pred / y_std, -1.0)

    return signal


def strategy_combined_best(data: dict, df: pd.DataFrame) -> pd.Series:
    """
    Combined strategy: weight the best individual signals.
    Use only signals that showed promise in individual backtests.
    """
    onds = data["onds"]
    signals = {}

    signals["peer"] = strategy_peer_leadlag(data) * 0.3
    signals["dix"] = strategy_dix_enhanced(data) * 0.3
    signals["regime"] = strategy_regime_conditional(data) * 0.2
    signals["vol_breakout"] = strategy_volatility_breakout(data) * 0.1
    signals["mean_rev"] = strategy_mean_reversion(data) * 0.1

    # Combine
    combined = pd.DataFrame(signals).sum(axis=1)
    return combined.clip(-1, 1)


# ═══════════════════════════════════════════════════════════════════
# SECTION 6: VOLATILITY-ADJUSTED POSITION SIZING
# ═══════════════════════════════════════════════════════════════════

def vol_adjusted_signal(raw_signal: pd.Series, prices: pd.Series,
                         target_vol: float = 0.30) -> pd.Series:
    """
    Scale position size inversely to recent volatility.
    In high-vol periods: reduce position.
    In low-vol periods: increase position.
    Target a constant annualized portfolio volatility.
    """
    ret = prices.pct_change()
    realized_vol = ret.rolling(20).std() * np.sqrt(252)
    vol_scalar = target_vol / realized_vol.clip(lower=0.05)
    vol_scalar = vol_scalar.clip(upper=2.0)  # max 2x leverage
    adjusted = raw_signal * vol_scalar
    return adjusted.clip(-1, 1)


# ═══════════════════════════════════════════════════════════════════
# SECTION 7: MAIN RUNNER
# ═══════════════════════════════════════════════════════════════════

def run_all_experiments():
    """Run every possible experiment and report results."""
    print("=" * 70)
    print("  ADVANCED QUANTITATIVE RESEARCH — ONDS")
    print("=" * 70)

    # ── Load Data ───────────────────────────────────────────────
    print("\n[1/7] Loading data...")
    data = load_all_data()

    # ── Build Features ──────────────────────────────────────────
    print("\n[2/7] Building advanced features...")
    df = build_advanced_features(data)
    df.to_csv(DATA_FEAT / "advanced_features.csv")

    # ── Feature Selection ───────────────────────────────────────
    print("\n[3/7] Feature selection...")
    targets_cls = ["target_direction", "target_intraday_dir", "target_gap_dir"]
    targets_reg = ["target_return", "target_abs_return", "target_range", "target_intraday"]

    # Exclude same-day features that would be data leakage for next-day prediction
    # Keep only features that are known at market close (lagged, rolling, peer lag, etc.)
    leakage_cols = {"ret_open_close", "ret_high_low", "ret_close_high", "ret_close_low",
                    "gap", "volume_change", "volume_ma_ratio", "dollar_volume", "price_range_pct",
                    "ret_1d", "dow", "is_monday", "is_friday"}
    all_feature_cols = [c for c in df.columns if not c.startswith("target_") and c not in leakage_cols]

    selected = {}
    for t in targets_cls + targets_reg:
        top = select_features(df, t, n_top=20, allowed_cols=all_feature_cols)
        selected[t] = top
        print(f"  {t}: top features = {top[:5]}...")

    # ── Model Evaluation: Classification ────────────────────────
    print("\n[4/7] Classification experiments...")
    print("-" * 70)

    cls_results = []

    # Experiment 1: Direction prediction with ALL features (limited to 40 to avoid overfit)
    r = evaluate_classification(df, all_feature_cols[:40], "target_direction", label="Direction (top-40 all)")
    if r: cls_results.append(r)

    # Experiment 2: Direction with SELECTED features
    r = evaluate_classification(df, selected["target_direction"], "target_direction",
                                 label="Direction (top-20 MI)")
    if r: cls_results.append(r)

    # Experiment 3: Direction with top-10
    top10 = selected["target_direction"][:10]
    r = evaluate_classification(df, top10, "target_direction", label="Direction (top-10 MI)")
    if r: cls_results.append(r)

    # Experiment 4: Intraday direction
    r = evaluate_classification(df, selected["target_intraday_dir"], "target_intraday_dir",
                                 label="Intraday dir (top-20)")
    if r: cls_results.append(r)

    # Experiment 5: Gap direction
    r = evaluate_classification(df, selected["target_gap_dir"], "target_gap_dir",
                                 label="Gap dir (top-20)")
    if r: cls_results.append(r)

    # Experiment 6: 3-class prediction
    r = evaluate_classification(df, selected["target_direction"], "target_3class",
                                 label="3-class (top-20)")
    if r: cls_results.append(r)

    # Experiment 7: Peer-only features → direction
    peer_feats = [c for c in all_feature_cols if any(p in c for p in ["RCAT", "JOBY", "AVAV", "KTOS", "LMT", "RTX", "peer"])]
    if len(peer_feats) >= 5:
        r = evaluate_classification(df, peer_feats, "target_direction", label="Direction (peer-only)")
        if r: cls_results.append(r)

    # Experiment 8: OHLCV-only features
    ohlcv_feats = [c for c in all_feature_cols if any(k in c for k in ["ret_", "gap", "range", "volume", "bb_", "rsi", "macd"])]
    if len(ohlcv_feats) >= 5:
        r = evaluate_classification(df, ohlcv_feats[:20], "target_direction", label="Direction (OHLCV-only)")
        if r: cls_results.append(r)

    # Print classification results
    print(f"\n{'='*70}")
    print("  CLASSIFICATION RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Experiment':<35s} {'Best Model':<8s} {'Accuracy':>10s} {'Std':>8s} {'N feat':>7s} {'N':>6s}")
    print(f"  {'-'*35} {'-'*8} {'-'*10} {'-'*8} {'-'*7} {'-'*6}")
    for r in cls_results:
        if r:
            std_str = f"+-{r['best_std']:.4f}"
            print(f"  {r['label']:<35s} {r['best_model']:<8s} {r['best_accuracy']:>10.4f} "
                  f"{std_str:>10s} {r['n_features']:>7d} {r['n_samples']:>6d}")
    print(f"  {'Baseline (random)':<35s} {'':8s} {'0.5000':>10s}")

    # ── Model Evaluation: Regression ────────────────────────────
    print(f"\n[5/7] Regression experiments...")
    print("-" * 70)

    reg_results = []

    # Return magnitude
    r = evaluate_regression(df, selected["target_return"], "target_return", label="Return magnitude (top-20)")
    if r: reg_results.append(r)

    # Volatility prediction
    r = evaluate_regression(df, selected["target_abs_return"], "target_abs_return", label="Volatility (top-20)")
    if r: reg_results.append(r)

    # Range prediction
    r = evaluate_regression(df, selected["target_range"], "target_range", label="Daily range (top-20)")
    if r: reg_results.append(r)

    # Intraday return
    r = evaluate_regression(df, selected["target_intraday"], "target_intraday", label="Intraday return (top-20)")
    if r: reg_results.append(r)

    # Return with top-10
    top10_ret = selected["target_return"][:10]
    r = evaluate_regression(df, top10_ret, "target_return", label="Return magnitude (top-10)")
    if r: reg_results.append(r)

    print(f"\n{'='*70}")
    print("  REGRESSION RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Experiment':<35s} {'Best':<8s} {'R2':>8s} {'Dir Acc':>10s} {'Std':>8s} {'N':>6s}")
    print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*10} {'-'*8} {'-'*6}")
    for r in reg_results:
        if r:
            std_str = f"+-{r['best_dir_acc_std']:.4f}"
            print(f"  {r['label']:<35s} {r['best_model']:<8s} {r['best_r2']:>8.4f} "
                  f"{r['best_dir_acc']:>10.4f} {std_str:>10s} {r['n_samples']:>6d}")

    # ── Strategy Backtests ──────────────────────────────────────
    print(f"\n[6/7] Strategy backtests...")
    print("-" * 70)

    onds_close = data["onds"]["Close"]
    all_bt_results = []

    strategies = [
        ("Peer Lead-Lag", strategy_peer_leadlag(data)),
        ("Market Contrarian", strategy_market_contrarian(data)),
        ("Regime Conditional", strategy_regime_conditional(data)),
        ("Vol Breakout", strategy_volatility_breakout(data)),
        ("DIX Enhanced", strategy_dix_enhanced(data)),
        ("Mean Reversion", strategy_mean_reversion(data)),
        ("Momentum+Stop", strategy_momentum_with_stop(data)),
        ("Gap Fade", strategy_gap_fade(data)),
        ("Volume Spike", strategy_volume_spike(data)),
        ("RSI Divergence", strategy_rsi_divergence(data)),
        ("Multi-TF Momentum", strategy_multi_timeframe_momentum(data)),
        ("BB Mean Reversion", strategy_bollinger_mean_reversion(data)),
        ("Combined Best", strategy_combined_best(data, df)),
        ("Adaptive Ensemble", strategy_adaptive_ensemble(data, df)),
    ]

    # Add vol-adjusted versions of top strategies
    for name, sig in strategies[:]:
        vol_sig = vol_adjusted_signal(sig, onds_close)
        strategies.append((f"{name} (VolAdj)", vol_sig))

    # ML signal with best features
    best_cls = selected["target_direction"][:15]
    ml_sig = strategy_ml_signal(df, best_cls, "target_direction")
    strategies.append(("ML Direction (top-15)", ml_sig))

    # Regression signal
    best_reg = selected["target_return"][:15]
    reg_sig = strategy_regression_signal(df, best_reg)
    strategies.append(("ML Regression Signal", reg_sig))

    for name, sig in strategies:
        try:
            r = backtest(onds_close, sig, name=name, plot=True, save_fig=True)
            print_results(r)
            all_bt_results.append(r)
        except Exception as e:
            print(f"  {name}: FAILED — {e}")

    # ── Comprehensive Comparison ────────────────────────────────
    print(f"\n[7/7] Final comparison...")
    print("=" * 70)

    if all_bt_results:
        comp_df = compare_strategies(all_bt_results)
        comp_df.to_csv(REPORTS_DIR / "advanced_strategy_comparison.csv", index=False)

    # Save summary
    summary = {
        "classification": [{k: v for k, v in r.items() if k != "feature_cols" and k != "all_results"} for r in cls_results if r],
        "regression": [{k: v for k, v in r.items() if k != "feature_cols" and k != "all_results"} for r in reg_results if r],
        "best_strategies": [],
    }

    # Rank strategies by Sharpe
    ranked = sorted(all_bt_results, key=lambda r: r["sharpe"], reverse=True)
    print(f"\n{'='*70}")
    print("  FINAL STRATEGY RANKING (by Sharpe Ratio)")
    print(f"{'='*70}")
    for i, r in enumerate(ranked):
        marker = " << BEST" if i == 0 else ""
        print(f"  {i+1:2d}. {r['name']:<30s} Sharpe={r['sharpe']:+.4f}  "
              f"Return={r['total_return']:+.2%}  MaxDD={r['max_drawdown']:.2%}  "
              f"Trades={r['n_trades']}{marker}")
        summary["best_strategies"].append({
            "rank": i+1, "name": r["name"], "sharpe": r["sharpe"],
            "total_return": r["total_return"], "max_drawdown": r["max_drawdown"],
            "hit_rate": r["hit_rate"], "n_trades": r["n_trades"],
        })

    pd.DataFrame(summary["best_strategies"]).to_csv(REPORTS_DIR / "strategy_ranking.csv", index=False)

    return summary, all_bt_results, cls_results, reg_results, df


if __name__ == "__main__":
    summary, bt_results, cls_results, reg_results, features = run_all_experiments()
