"""
Plan 12: Multi-Source Signal Fusion

Combines all available signals into a unified model.
Tests feature importance and builds an ensemble predictor.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_PROC, DATA_FEAT, FIGURES_DIR


def build_feature_matrix(
    onds_prices: pd.DataFrame,
    technical_df: pd.DataFrame = None,
    crossasset_features: pd.DataFrame = None,
    sector_features: pd.DataFrame = None,
    reddit_sentiment: pd.DataFrame = None,
    news_sentiment: pd.DataFrame = None,
    darkpool_signal: pd.Series = None,
    regime_df: pd.DataFrame = None,
    analyst_signal: pd.Series = None,
) -> pd.DataFrame:
    """
    Merge all feature sources into a single feature matrix aligned by date.
    Target: next-day ONDS return direction (1 = up, 0 = down).
    """
    # Start with ONDS returns
    features = pd.DataFrame(index=onds_prices.index)
    features["return_1d"] = onds_prices["Close"].pct_change()
    features["target"] = (features["return_1d"].shift(-1) > 0).astype(int)  # next-day direction

    # Technical features
    if technical_df is not None:
        tech_cols = [c for c in technical_df.columns
                     if c not in ["Open", "High", "Low", "Close", "Volume", "Fwd_1d", "Fwd_5d"]]
        for col in tech_cols:
            if col in technical_df.columns:
                features[f"tech_{col}"] = technical_df[col]

    # Cross-asset features
    if crossasset_features is not None:
        for col in crossasset_features.columns:
            if col != "ONDS_ret_1d":
                features[f"xa_{col}"] = crossasset_features[col]

    # Sector features
    if sector_features is not None:
        for col in sector_features.columns:
            if "ONDS" not in col:
                features[f"sec_{col}"] = sector_features[col]

    # Reddit sentiment
    if reddit_sentiment is not None and not reddit_sentiment.empty:
        for col in reddit_sentiment.columns:
            features[f"reddit_{col}"] = reddit_sentiment[col]

    # News sentiment
    if news_sentiment is not None and not news_sentiment.empty:
        for col in news_sentiment.columns:
            features[f"news_{col}"] = news_sentiment[col]

    # Dark pool
    if darkpool_signal is not None and not darkpool_signal.empty:
        features["darkpool_signal"] = darkpool_signal

    # Regime
    if regime_df is not None and not regime_df.empty:
        if "regime" in regime_df.columns:
            features["regime"] = regime_df["regime"]

    # Analyst
    if analyst_signal is not None and not analyst_signal.empty:
        features["analyst_signal"] = analyst_signal

    # Forward-fill NaN from low-frequency sources
    features = features.ffill().bfill()

    print(f"  Feature matrix: {features.shape[0]} rows × {features.shape[1]} columns")
    print(f"  Non-null features: {features.notna().sum().sum()}")
    print(f"  Feature columns: {list(features.columns[:10])}... ({len(features.columns)} total)")

    return features


def train_ensemble_model(
    features: pd.DataFrame,
    n_splits: int = 5,
    save: bool = True,
) -> dict:
    """
    Train time-series cross-validated ensemble model.
    Uses walk-forward validation (no future leakage).
    """
    # Separate target
    target_col = "target"
    if target_col not in features.columns:
        print("  No target column. Cannot train model.")
        return {}

    # Remove non-feature columns
    drop_cols = [target_col, "return_1d"]
    feature_cols = [c for c in features.columns if c not in drop_cols]

    # Drop columns that are >50% NaN, then fill remaining NaN with 0
    df = features[feature_cols + [target_col]].copy()
    null_pct = df[feature_cols].isnull().mean()
    keep_cols = null_pct[null_pct < 0.5].index.tolist()
    feature_cols = [c for c in keep_cols if c != target_col]
    df = df[feature_cols + [target_col]]
    df[feature_cols] = df[feature_cols].ffill().fillna(0)
    df = df.dropna(subset=[target_col])
    X = df[feature_cols]
    y = df[target_col]

    print(f"\n  Training ensemble model...")
    print(f"    Features: {len(feature_cols)}")
    print(f"    Samples: {len(X)}")
    print(f"    Class balance: {y.mean():.2%} positive")

    # Time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scaler = StandardScaler()

    rf_scores = []
    gb_scores = []
    feature_importances = np.zeros(len(feature_cols))

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Random Forest
        rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
        rf.fit(X_train_s, y_train)
        rf_pred = rf.predict(X_test_s)
        rf_scores.append(accuracy_score(y_test, rf_pred))
        feature_importances += rf.feature_importances_

        # Gradient Boosting
        gb = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42)
        gb.fit(X_train_s, y_train)
        gb_pred = gb.predict(X_test_s)
        gb_scores.append(accuracy_score(y_test, gb_pred))

    feature_importances /= n_splits

    results = {
        "rf_accuracy_mean": np.mean(rf_scores),
        "rf_accuracy_std": np.std(rf_scores),
        "gb_accuracy_mean": np.mean(gb_scores),
        "gb_accuracy_std": np.std(gb_scores),
        "feature_importance": dict(zip(feature_cols, feature_importances)),
        "n_features": len(feature_cols),
        "n_samples": len(X),
    }

    print(f"\n  Ensemble Results (Time-Series CV, {n_splits} folds):")
    print(f"    Random Forest:     {results['rf_accuracy_mean']:.4f} ± {results['rf_accuracy_std']:.4f}")
    print(f"    Gradient Boosting: {results['gb_accuracy_mean']:.4f} ± {results['gb_accuracy_std']:.4f}")
    print(f"    Baseline (random): 0.5000")

    # Top features
    sorted_imp = sorted(results["feature_importance"].items(), key=lambda x: -x[1])
    print(f"\n  Top 15 Features:")
    for feat, imp in sorted_imp[:15]:
        print(f"    {feat:40s} {imp:.4f}")

    if save:
        imp_df = pd.DataFrame(sorted_imp, columns=["Feature", "Importance"])
        imp_df.to_csv(DATA_PROC / "feature_importance.csv", index=False)
        _plot_feature_importance(sorted_imp[:20])

    return results


def _plot_feature_importance(sorted_imp: list, save: bool = True):
    """Plot top features by importance."""
    features, importances = zip(*sorted_imp)

    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = range(len(features))
    ax.barh(y_pos, importances, color="#1976D2", alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance (RF)")
    ax.set_title("Top Features for ONDS Return Prediction")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_fusion_signal(features: pd.DataFrame) -> pd.Series:
    """
    Generate a combined signal by training on historical data
    and predicting the latest period.
    Uses expanding window: train on all data up to t, predict t+1.
    """
    target_col = "target"
    drop_cols = [target_col, "return_1d"]
    feature_cols = [c for c in features.columns if c not in drop_cols]

    df = features[feature_cols + [target_col]].copy()
    null_pct = df[feature_cols].isnull().mean()
    keep_cols = null_pct[null_pct < 0.5].index.tolist()
    feature_cols = [c for c in keep_cols if c != target_col]
    df = df[feature_cols + [target_col]]
    df[feature_cols] = df[feature_cols].ffill().fillna(0)
    df = df.dropna(subset=[target_col])
    X = df[feature_cols]
    y = df[target_col]

    scaler = StandardScaler()
    signal = pd.Series(0.0, index=df.index)

    # Walk-forward: train on expanding window, predict next day
    min_train = 60
    for i in range(min_train, len(X)):
        X_train = scaler.fit_transform(X.iloc[:i])
        y_train = y.iloc[:i]
        X_test = scaler.transform(X.iloc[i:i+1])

        rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
        rf.fit(X_train, y_train)
        prob = rf.predict_proba(X_test)[0]

        # Convert probability to signal: P(up) > 0.55 → long, P(up) < 0.45 → short
        if len(prob) == 2:
            p_up = prob[1]
            if p_up > 0.55:
                signal.iloc[i] = 1.0
            elif p_up < 0.45:
                signal.iloc[i] = -1.0

    return signal
