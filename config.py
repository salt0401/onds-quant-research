"""
Central configuration for the ONDS Quantitative Research project.
All paths, tickers, API keys, and parameters in one place.
"""
import os
from pathlib import Path

# Load .env file if present (API keys, credentials)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass  # python-dotenv not installed, use os.environ directly

# ── Project root ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_RAW     = PROJECT_ROOT / "data" / "raw"
DATA_PROC    = PROJECT_ROOT / "data" / "processed"
DATA_FEAT    = PROJECT_ROOT / "data" / "features"
OUTPUT_DIR   = PROJECT_ROOT / "output"
FIGURES_DIR  = OUTPUT_DIR / "figures"
REPORTS_DIR  = OUTPUT_DIR / "reports"

for d in [DATA_RAW, DATA_PROC, DATA_FEAT, FIGURES_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Target & Universe ─────────────────────────────────────────
TARGET_TICKER = "ONDS"

# Related drone / defense stocks
PEER_TICKERS = ["RCAT", "AVAV", "KTOS", "JOBY", "LMT", "RTX"]

# Sector ETFs
SECTOR_ETFS = ["ITA", "XAR"]      # iShares US Aerospace & Defense, SPDR S&P Aerospace

# Cross-asset
CROSS_ASSET = {
    "gold":    "GLD",
    "silver":  "SLV",
    "bitcoin": "BTC-USD",
    "vix":     "^VIX",
    "sp500":   "SPY",
    "nasdaq":  "QQQ",
    "bonds":   "TLT",
    "usd":     "UUP",
}

# All tickers to download
ALL_TICKERS = (
    [TARGET_TICKER]
    + PEER_TICKERS
    + SECTOR_ETFS
    + list(CROSS_ASSET.values())
)

# ── Date range ────────────────────────────────────────────────
DATA_START = "2020-01-01"
DATA_END   = "2026-02-15"

# ── API Keys (set via environment variables) ──────────────────
FINNHUB_API_KEY   = os.getenv("FINNHUB_API_KEY", "")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "")
REDDIT_CLIENT_ID  = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_SECRET     = os.getenv("REDDIT_SECRET", "")

# ── Technical Analysis Parameters ─────────────────────────────
TA_PARAMS = {
    "rsi_period":       14,
    "macd_fast":        12,
    "macd_slow":        26,
    "macd_signal":      9,
    "bb_period":        20,
    "bb_std":           2,
    "atr_period":       14,
    "obv_ma":           20,
    "sma_short":        10,
    "sma_long":         50,
}

# ── Backtest Parameters ───────────────────────────────────────
BACKTEST = {
    "initial_capital":  100_000,
    "commission_bps":   10,        # 10 bps round-trip
    "slippage_bps":     5,
    "risk_free_rate":   0.05,      # annualized
}

# ── Regime Detection Parameters ───────────────────────────────
REGIME = {
    "n_regimes":        3,         # bull / bear / high-vol
    "hmm_covariance":   "full",
    "lookback_window":  252,       # 1 year
}

# ── Sentiment Parameters ──────────────────────────────────────
SENTIMENT = {
    "reddit_subreddits": ["ONDS", "Ondasholdings", "pennystocks", "wallstreetbets"],
    "reddit_limit":      1000,
    "finbert_model":     "ProsusAI/finbert",
    "vader_threshold":   0.05,
}
