# ONDS Multi-Signal Quantitative Research

A comprehensive quantitative research framework analyzing **Ondas Holdings (ONDS)** from 12 different signal perspectives. Combines NLP sentiment analysis (Reddit, news), technical indicators, cross-asset signals (gold, silver, bitcoin, VIX), dark pool data, options/IV analysis, sector analysis (drone/defense peers), event studies, and regime detection into a unified research pipeline.

## Key Findings

| Strategy | Sharpe | Annual Return | Max Drawdown | Hit Rate |
|----------|--------|--------------|--------------|----------|
| Technical Composite | +0.21 | +23.1% | -67.6% | 52.3% |
| Cross-Asset | -0.34 | +3.0% | -10.0% | 57.9% |
| Sector Momentum | -0.39 | -9.0% | -54.6% | 50.5% |
| Reddit Sentiment | * | * | -47.0% | 56.6% |
| News Sentiment | -0.12 | -12.2% | -62.1% | 57.7% |
| Multi-Source Fusion (RF) | -0.62 | -49.2% | -98.6% | 53.7% |

\*Reddit sentiment has limited data (80 trading days) — results not reliable.

### Notable Results
- **Technical MACD signal**: Sharpe +0.34, total return +546% (but with 87% max drawdown)
- **Regime detection**: HMM identifies 3 clear regimes — Bearish (40%, mean=-1.2%/day), Neutral (41%), Bullish (19%, mean=+4.6%/day)
- **Event study**: Acquisition events show significant CAR (+114%, p=0.003)
- **Options**: ATM IV ~117%, put-call volume ratio 0.49 (bullish), IV term structure slightly inverted
- **Cross-asset**: Bond returns (TLT 20d) are the only significant predictor (p=0.049) of ONDS next-day returns
- **Top ML features**: Peer average return, NASDAQ 5d return, bond 20d return, RCAT 5d return
- **RCAT lead-lag**: Significant negative lag-2 correlation (r=-0.11, p=0.014)

## Pipeline Architecture

```
Data Collection (Plan 0)
├── yfinance: ONDS + 16 related tickers (1,300+ trading days)
├── Reddit JSON API: 362 posts from 7 subreddits
├── Google News RSS: 100+ articles
├── yfinance Options: 529 contracts, 10 expirations
└── Dark pool: SqueezeMetrics DIX/GEX, Stockgrid (manual CSV)

Analysis (Plans 1-11)
├── Plan 1:  Technical Analysis (RSI, MACD, Bollinger, ATR, OBV, SMA)
├── Plan 2:  Sector Analysis (RCAT, AVAV, KTOS, JOBY, LMT, RTX, ITA)
├── Plan 3:  Cross-Asset (GLD, SLV, BTC, VIX, SPY, QQQ, TLT, UUP)
├── Plan 4:  Dark Pool (DIX/GEX, Stockgrid net short volume)
├── Plan 5:  Reddit Sentiment (VADER + FinBERT ensemble)
├── Plan 6:  News Sentiment (Google News + NLP scoring)
├── Plan 7:  CEO Tweet Analysis (framework — needs X credentials)
├── Plan 8:  Analyst Reports (framework — needs Finnhub API key)
├── Plan 9:  Event Studies (12 known ONDS events, CAR analysis)
├── Plan 10: Options & IV Surface (smile, surface, skew, term structure)
└── Plan 11: Regime Detection (HMM, GMM, Markov Switching, volatility-based)

Fusion (Plan 12)
├── 82-feature matrix (tech + cross-asset + sector + sentiment + regime)
├── Random Forest & Gradient Boosting (time-series CV)
├── Feature importance ranking
└── Walk-forward backtest
```

## Directory Structure

```
ONDS_Research/
├── config.py                    # Central configuration (tickers, parameters)
├── run_all.py                   # Master pipeline orchestrator
├── requirements.txt
├── collectors/                  # Data collection modules
│   ├── prices.py               # yfinance price data
│   ├── reddit.py               # Reddit JSON API scraper
│   ├── news.py                 # Google News + Finnhub
│   ├── darkpool.py             # SqueezeMetrics + Stockgrid
│   └── options.py              # yfinance options chains
├── analysis/                    # Analysis modules
│   ├── technical.py            # Technical indicators + predictive power
│   ├── crossasset.py           # Cross-asset correlations + signals
│   ├── sector.py               # Peer/sector lead-lag + momentum
│   ├── darkpool.py             # DIX/GEX analysis
│   ├── sentiment.py            # VADER + FinBERT sentiment scoring
│   ├── analyst.py              # Analyst recommendation analysis
│   ├── events.py               # Event study methodology
│   ├── options_iv.py           # IV smile, surface, derived features
│   ├── regime.py               # HMM, GMM, Markov Switching
│   └── fusion.py               # Multi-source feature matrix + ML
├── backtests/
│   └── engine.py               # Shared backtest framework
├── data/
│   ├── raw/                    # Downloaded data (not in repo)
│   ├── processed/              # Analyzed data
│   └── features/               # ML feature matrices
└── output/
    ├── figures/                # 18 generated plots
    └── reports/                # Strategy comparison CSV
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline (downloads data + runs all analyses)
python run_all.py

# Run with cached data (skip downloads)
python run_all.py --skip-collect

# Run specific plans only
python run_all.py --skip-collect --plan 1 3 11 12
```

### Optional API Keys (for enhanced data)
```bash
export FINNHUB_API_KEY="your_key"        # Free at finnhub.io
export ALPHA_VANTAGE_KEY="your_key"      # Free at alphavantage.co
export REDDIT_CLIENT_ID="your_id"        # For PRAW (enhanced Reddit)
export REDDIT_SECRET="your_secret"
```

## Generated Outputs

### Figures (18 plots)
- `onds_technical_dashboard.png` — Price + RSI + MACD + Volume
- `sector_comparison.png` — ONDS vs drone/defense peers (normalized)
- `crossasset_correlation_heatmap.png` — Full correlation matrix
- `onds_iv_smile.png` — IV smile across expirations
- `onds_iv_surface.png` — 3D IV surface
- `regime_hmm.png` / `regime_volatility.png` / `regime_markovswitching.png` — Regime overlays
- `sentiment_reddit_vs_price.png` / `sentiment_news_vs_price.png`
- `feature_importance.png` — Top ML features
- `strategy_comparison.png` — All strategies overlaid
- 6x backtest equity curves (`bt_*.png`)

### Data Files
- `technical_predictive_power.csv` — Spearman correlations for each indicator
- `crossasset_predictive_power.csv` — Cross-asset predictive tests
- `sector_lead_lag.csv` — Peer lead-lag analysis
- `event_study_results.csv` — CAR for 12 ONDS events
- `reddit_daily_sentiment.csv` / `news_daily_sentiment.csv`
- `regimes_hmm.csv` / `regimes_markov.csv` / `regimes_volatility.csv`
- `feature_importance.csv` — ML feature rankings
- `strategy_comparison.csv` — Final strategy comparison table

## Research Methodology

### Sentiment Analysis
- **VADER**: Rule-based, handles social media slang well
- **FinBERT**: Transformer-based, domain-specific for finance (when available)
- **Ensemble**: Average of both scores, z-score normalized for signals

### Regime Detection
- **HMM (3 regimes)**: Gaussian emissions, identifies bull/bear/neutral periods
- **Markov Switching (2 regimes)**: Hamilton (1989) model with switching variance
- **Volatility-based**: Rolling 20-day annualized vol, quantile thresholds

### Backtesting
- **Lag-1 execution**: Signal on day t → position on day t+1
- **Transaction costs**: 10 bps round-trip
- **Metrics**: Sharpe, annual return/vol, max drawdown, hit rate
- **Walk-forward**: Expanding window for ML predictions (no future leakage)

## ONDS Company Profile

**Ondas Holdings (NASDAQ: ONDS)** — Defense/drone technology company
- Market cap: ~$4.37B (as of early 2026)
- Products: AURA drone system, FullMAX wireless network
- Subsidiary: American Robotics (autonomous drone solutions)
- Key catalysts: DoD contracts, NATO partnerships, FAA drone regulations
- CEO: Eric Brock (@CeoOndas on X)
- Peer universe: RCAT, AVAV, KTOS, JOBY, LMT, RTX

## Future Improvements

### CEO Twitter/X Sentiment (Plan 7)
- **Goal**: Scrape tweets from @CeoOndas (Eric Brock) and analyze sentiment around key announcements
- **Tool**: `twscrape` (installed) — requires X account authentication
- **Setup**: Run `python -m twscrape add_accounts "username" "password" "email" "email_password"` then `python -m twscrape login_accounts`
- **Alternative**: Apply for X API developer access at [developer.twitter.com](https://developer.twitter.com)
- **Analysis**: Score tweets with FinBERT, compute event-window returns around tweet bursts
- **Status**: Framework built (`analysis/sentiment.py` supports Twitter), awaiting authentication

### Additional Enhancements
- **Real-time dark pool**: Integrate live SqueezeMetrics DIX/GEX feed (requires subscription)
- **Intraday options flow**: Monitor unusual options activity via CBOE/OPRA data
- **Insider transactions**: Track SEC Form 4 filings for ONDS insiders (via SEC EDGAR API)
- **Reddit real-time**: Use PRAW (Reddit API) for streaming sentiment instead of JSON API snapshots
- **Ensemble tuning**: Hyperparameter optimization (Optuna), SHAP feature explanations, neural network models
- **Portfolio optimization**: Mean-variance or risk-parity allocation across multiple signal strategies

## License

Academic and research use. Part of a graduation thesis on multi-signal quantitative analysis.
