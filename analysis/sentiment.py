"""
Plans 5/6/7: Unified Sentiment Analysis Module

Applies VADER (lightweight) and optionally FinBERT (transformer-based)
to Reddit posts, news headlines, and tweets. Aggregates to daily sentiment.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_RAW, DATA_PROC, FIGURES_DIR

# Lazy-load sentiment models
_vader = None
_finbert = None


def get_vader():
    """Lazy-load VADER sentiment analyzer."""
    global _vader
    if _vader is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _vader = SentimentIntensityAnalyzer()
    return _vader


def get_finbert():
    """Lazy-load FinBERT pipeline."""
    global _finbert
    if _finbert is None:
        try:
            from transformers import pipeline
            _finbert = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                truncation=True,
                max_length=512,
            )
            print("  FinBERT loaded successfully")
        except Exception as e:
            print(f"  FinBERT failed to load: {e}")
            print("  Falling back to VADER only")
            _finbert = "unavailable"
    return _finbert if _finbert != "unavailable" else None


def score_vader(text: str) -> float:
    """Score text with VADER. Returns compound score in [-1, +1]."""
    if not text or not isinstance(text, str):
        return 0.0
    return get_vader().polarity_scores(text)["compound"]


def score_finbert(texts: list[str], batch_size: int = 32) -> list[float]:
    """
    Score texts with FinBERT. Returns list of scores in [-1, +1].
    Maps: positive → +1, negative → -1, neutral → 0, scaled by confidence.
    """
    pipe = get_finbert()
    if pipe is None:
        return [score_vader(t) for t in texts]

    scores = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        # Clean empty strings
        batch = [t if t and isinstance(t, str) else "neutral" for t in batch]
        try:
            results = pipe(batch)
            for r in results:
                label = r["label"].lower()
                conf = r["score"]
                if label == "positive":
                    scores.append(conf)
                elif label == "negative":
                    scores.append(-conf)
                else:
                    scores.append(0.0)
        except Exception:
            scores.extend([score_vader(t) for t in batch])
    return scores


def analyze_reddit_sentiment(save: bool = True) -> pd.DataFrame:
    """
    Load Reddit posts, compute sentiment, aggregate to daily.
    """
    path = DATA_RAW / "reddit_all_onds_posts.csv"
    if not path.exists():
        print("  No Reddit data. Run collectors/reddit.py first.")
        return pd.DataFrame()

    df = pd.read_csv(path)
    print(f"  Analyzing sentiment for {len(df)} Reddit posts...")

    # Combine title + selftext for scoring
    df["text"] = df["title"].fillna("") + " " + df["selftext"].fillna("")
    df["text"] = df["text"].str.strip()

    # VADER scores
    df["vader_score"] = df["text"].apply(score_vader)

    # FinBERT scores (if available)
    texts = df["text"].tolist()
    df["finbert_score"] = score_finbert(texts)

    # Ensemble: average of VADER and FinBERT
    df["sentiment"] = (df["vader_score"] + df["finbert_score"]) / 2

    # Parse dates
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    elif "created_utc" in df.columns:
        df["date"] = pd.to_datetime(df["created_utc"], unit="s").dt.date
        df["date"] = pd.to_datetime(df["date"])

    # Daily aggregation
    daily = df.groupby("date").agg(
        n_posts=("sentiment", "count"),
        mean_sentiment=("sentiment", "mean"),
        median_sentiment=("sentiment", "median"),
        std_sentiment=("sentiment", "std"),
        mean_score=("score", "mean"),
        total_comments=("num_comments", "sum"),
        vader_mean=("vader_score", "mean"),
        finbert_mean=("finbert_score", "mean"),
    ).sort_index()

    # Volume-weighted sentiment (weight by post score)
    if "score" in df.columns:
        df["abs_score"] = df["score"].abs().clip(lower=1)
        weighted = df.groupby("date").apply(
            lambda g: np.average(g["sentiment"], weights=g["abs_score"])
            if len(g) > 0 else 0
        )
        daily["weighted_sentiment"] = weighted

    if save:
        df.to_csv(DATA_PROC / "reddit_posts_scored.csv", index=False)
        daily.to_csv(DATA_PROC / "reddit_daily_sentiment.csv")
        print(f"  Saved scored posts and daily sentiment to {DATA_PROC}")

    print(f"\n  Reddit Sentiment Summary:")
    print(f"    Posts scored: {len(df)}")
    print(f"    Date range: {daily.index.min()} to {daily.index.max()}")
    print(f"    Mean VADER:   {df['vader_score'].mean():+.4f}")
    print(f"    Mean FinBERT: {df['finbert_score'].mean():+.4f}")
    print(f"    Mean Ensemble:{df['sentiment'].mean():+.4f}")

    return daily


def analyze_news_sentiment(save: bool = True) -> pd.DataFrame:
    """Load news articles, compute sentiment, aggregate to daily."""
    path = DATA_RAW / "google_news_onds.csv"
    if not path.exists():
        print("  No news data. Run collectors/news.py first.")
        return pd.DataFrame()

    df = pd.read_csv(path)
    print(f"  Analyzing sentiment for {len(df)} news articles...")

    df["text"] = df["title"].fillna("")
    df["vader_score"] = df["text"].apply(score_vader)
    df["finbert_score"] = score_finbert(df["text"].tolist())
    df["sentiment"] = (df["vader_score"] + df["finbert_score"]) / 2

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    elif "datetime" in df.columns:
        df["date"] = pd.to_datetime(df["datetime"]).dt.date
        df["date"] = pd.to_datetime(df["date"])

    daily = df.groupby("date").agg(
        n_articles=("sentiment", "count"),
        mean_sentiment=("sentiment", "mean"),
        vader_mean=("vader_score", "mean"),
        finbert_mean=("finbert_score", "mean"),
    ).sort_index()

    if save:
        df.to_csv(DATA_PROC / "news_articles_scored.csv", index=False)
        daily.to_csv(DATA_PROC / "news_daily_sentiment.csv")

    print(f"\n  News Sentiment Summary:")
    print(f"    Articles scored: {len(df)}")
    print(f"    Mean sentiment: {df['sentiment'].mean():+.4f}")

    return daily


def test_sentiment_predictive_power(
    daily_sentiment: pd.DataFrame,
    onds_prices: pd.DataFrame,
    source_name: str = "Reddit",
    save: bool = True,
) -> pd.DataFrame:
    """
    Test if daily sentiment predicts ONDS returns.
    Uses Spearman correlation at various lags.
    """
    onds_ret = onds_prices["Close"].pct_change()

    results = []
    sent_cols = [c for c in daily_sentiment.columns if "sentiment" in c.lower()]

    for col in sent_cols:
        for lag in [0, 1, 2, 5]:
            merged = pd.DataFrame({
                "sentiment": daily_sentiment[col],
                "onds_fwd": onds_ret.shift(-lag),
            }).dropna()

            if len(merged) < 20:
                continue

            corr, pval = stats.spearmanr(merged["sentiment"], merged["onds_fwd"])
            results.append({
                "Source": source_name,
                "Sentiment_col": col,
                "Lag": lag,
                "Spearman": round(corr, 4),
                "p_value": round(pval, 4),
                "sig": "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else "",
                "n": len(merged),
            })

    df = pd.DataFrame(results)
    if save and not df.empty:
        df.to_csv(DATA_PROC / f"{source_name.lower()}_predictive_power.csv", index=False)

    print(f"\n  {source_name} Sentiment → ONDS Return Predictive Power:")
    print("  " + "-" * 60)
    for _, row in df.iterrows():
        print(f"    {row['Sentiment_col']:25s} lag={row['Lag']}  "
              f"r={row['Spearman']:+.4f}  p={row['p_value']:.4f} {row['sig']}")

    return df


def generate_sentiment_signal(daily_sentiment: pd.DataFrame) -> pd.Series:
    """Generate trading signal from daily sentiment."""
    if daily_sentiment.empty:
        return pd.Series(dtype=float)

    # Use mean_sentiment if available, else first sentiment column
    if "mean_sentiment" in daily_sentiment.columns:
        raw = daily_sentiment["mean_sentiment"]
    else:
        sent_cols = [c for c in daily_sentiment.columns if "sentiment" in c.lower()]
        raw = daily_sentiment[sent_cols[0]] if sent_cols else pd.Series(dtype=float)

    if raw.empty:
        return pd.Series(dtype=float)

    # Z-score normalization
    z = (raw - raw.rolling(20, min_periods=5).mean()) / raw.rolling(20, min_periods=5).std()

    # Signal: z > 0.5 → long, z < -0.5 → short
    sig = pd.Series(0.0, index=z.index)
    sig[z > 0.5] = 1.0
    sig[z < -0.5] = -1.0

    return sig


def plot_sentiment_vs_price(daily_sentiment: pd.DataFrame, onds_prices: pd.DataFrame,
                            source_name: str = "Reddit", save: bool = True):
    """Plot sentiment time series against ONDS price."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                                         height_ratios=[2, 1, 1])

    # Price
    ax1.plot(onds_prices.index, onds_prices["Close"], color="#1976D2", linewidth=1)
    ax1.set_ylabel("ONDS Price ($)")
    ax1.set_title(f"{source_name} Sentiment vs ONDS Price")
    ax1.grid(True, alpha=0.3)

    # Sentiment
    if "mean_sentiment" in daily_sentiment.columns:
        colors = ["#4CAF50" if v >= 0 else "#F44336" for v in daily_sentiment["mean_sentiment"]]
        ax2.bar(daily_sentiment.index, daily_sentiment["mean_sentiment"],
                color=colors, alpha=0.6, width=1)
        ax2.set_ylabel("Mean Sentiment")
        ax2.axhline(0, color="black", linewidth=0.5)
        ax2.grid(True, alpha=0.3)

    # Volume / count
    count_col = "n_posts" if "n_posts" in daily_sentiment.columns else "n_articles"
    if count_col in daily_sentiment.columns:
        ax3.bar(daily_sentiment.index, daily_sentiment[count_col],
                color="#9E9E9E", alpha=0.6, width=1)
        ax3.set_ylabel(f"# {source_name} Items")
        ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / f"sentiment_{source_name.lower()}_vs_price.png",
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
