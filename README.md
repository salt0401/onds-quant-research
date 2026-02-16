# ONDS Multi-Signal Quantitative Research

A comprehensive quantitative research framework analyzing **Ondas Holdings (ONDS)** from 12 different signal perspectives. Combines NLP sentiment analysis (Reddit, news), technical indicators, cross-asset signals (gold, silver, bitcoin, VIX), dark pool data, options/IV analysis, sector analysis (drone/defense peers), event studies, and regime detection into a unified research pipeline.

## Key Findings

### Advanced Strategy Results (30 strategies, 2025-01 to 2026-02)

| Strategy | Sharpe | Return | Max DD | Hit Rate | Perm p-value |
|----------|--------|--------|--------|----------|-------------|
| **ML Direction (top-15)** | **+4.06** | +314% | -26% | 61.8% | **0.009** |
| **Momentum+Stop (VolAdj)** | **+2.39** | +61% | -11% | 54.4% | **0.021** |
| **Adaptive Ensemble** | **+1.79** | +71% | -22% | 54.6% | **0.039** |
| Market Contrarian | +1.24 | +88% | -35% | 58.5% | - |
| Market Contrarian (VolAdj) | +1.11 | +24% | -8% | 58.5% | 0.064 |
| Combined Best | +1.09 | +41% | -18% | 52.9% | - |
| Multi-TF Momentum (VolAdj) | +1.02 | +34% | -16% | 50.9% | - |

**6 strategies with Sharpe > 1.0**, 3 statistically significant at 5% level via permutation test (2000 permutations).

### Robustness Validation

| Strategy | Bootstrap Sharpe | 95% CI | P(Sharpe>0) |
|----------|-----------------|--------|-------------|
| ML Direction (top-15) | +2.20 | [+0.39, +3.85] | 98.9% |
| Momentum+Stop (VolAdj) | +1.93 | [-0.21, +3.86] | 96.3% |
| Adaptive Ensemble | +1.49 | [-0.33, +3.28] | 94.5% |
| Market Contrarian (VolAdj) | +1.04 | [-0.78, +2.85] | 87.3% |

The ML Direction strategy's 95% CI is entirely positive with 94.4% of rolling 60-day windows showing positive Sharpe.

### Base Analysis Results (12 plans)

| Strategy | Sharpe | Annual Return | Max Drawdown | Hit Rate |
|----------|--------|--------------|--------------|----------|
| Technical Composite | +0.38 | +41.9% | -45.6% | 52.8% |
| Dark Pool (DIX/GEX) | +0.69 | +34.5% | -38.4% | 53.1% |
| Reddit Sentiment | +4.10 | +779%\* | -33.7% | 55.6% |
| Cross-Asset | -0.99 | -1.1% | -5.4% | 37.5% |
| Sector Momentum | -1.14 | -41.9% | -54.6% | 48.3% |

\*Reddit sentiment has limited data (46 trades, 80 trading days) — results not reliable.

### Notable Results
- **Peer lead-lag**: JOBY lag-1 significant (p=0.007), top ML feature for direction prediction
- **Regime detection**: HMM identifies 3 regimes — Bearish (86%), Neutral (8%), Bullish (6%, mean=+14.3%/day)
- **Event study**: Acquisition events show significant CAR (+114%, p=0.003)
- **Options**: ATM IV ~117%, put-call volume ratio 0.49 (bullish), IV term structure slightly inverted
- **Cross-asset**: Gold lag-1 and VIX change are significant direction predictors
- **Top ML features**: gold_ret_lag1, JOBY_ret_lag2, KTOS_ret_lag5, RCAT_ret_lag3, regime
- **Direction classification**: Peer-only features achieve 56.9% accuracy (above 50% baseline)
- **Gap direction**: 59.1% accuracy using lagged returns and peer features

## Pipeline Architecture

```
Data Collection (Plan 0)
├── yfinance: ONDS + 16 related tickers (280 trading days, 2025-01+)
├── Reddit JSON API: 362 posts from 7 subreddits
├── Google News RSS: 100+ articles
├── Finnhub API: 217 news articles + analyst recommendations
├── ChartExchange: ONDS FINRA short volume (375 days)
├── yfinance Options: 529 contracts, 10 expirations
└── Dark pool: SqueezeMetrics DIX/GEX (3,720 days)

Analysis (Plans 1-11)
├── Plan 1:  Technical Analysis (RSI, MACD, Bollinger, ATR, OBV, SMA)
├── Plan 2:  Sector Analysis (RCAT, AVAV, KTOS, JOBY, LMT, RTX, ITA)
├── Plan 3:  Cross-Asset (GLD, SLV, BTC, VIX, SPY, QQQ, TLT, UUP)
├── Plan 4:  Dark Pool (DIX/GEX + ChartExchange FINRA short volume)
├── Plan 5:  Reddit Sentiment (VADER + FinBERT ensemble)
├── Plan 6:  News Sentiment (Google News + NLP scoring)
├── Plan 7:  CEO Tweet Analysis (framework — needs X credentials)
├── Plan 8:  Analyst Reports (Finnhub: consensus 1.29, very bullish)
├── Plan 9:  Event Studies (12 known ONDS events, CAR analysis)
├── Plan 10: Options & IV Surface (smile, surface, skew, term structure)
└── Plan 11: Regime Detection (HMM, GMM, Markov Switching, volatility-based)

Fusion (Plan 12)
├── 103-feature matrix (tech + cross-asset + sector + sentiment + regime + short vol)
├── Random Forest & Gradient Boosting (time-series CV)
├── Feature importance ranking
└── Walk-forward backtest

Advanced Research (Plan 13)
├── 106 features with leakage-safe engineering
├── 8 classification experiments (direction, gap, intraday, 3-class)
├── 5 regression experiments (return, volatility, range)
├── 15 base strategies + 15 volatility-adjusted variants (30 total)
├── ML walk-forward signals (RF, GB, XGB, LR)
├── Bootstrap confidence intervals (5000 block-bootstrap samples)
├── Permutation tests (2000 shuffles)
└── Rolling Sharpe stability analysis
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
│   ├── darkpool.py             # DIX/GEX + ChartExchange short volume
│   ├── sentiment.py            # VADER + FinBERT sentiment scoring
│   ├── analyst.py              # Analyst recommendation analysis
│   ├── events.py               # Event study methodology
│   ├── options_iv.py           # IV smile, surface, derived features
│   ├── regime.py               # HMM, GMM, Markov Switching
│   ├── fusion.py               # Multi-source feature matrix + ML
│   ├── advanced_research.py    # 30 strategies, ML signals, feature selection
│   └── robustness.py           # Bootstrap, permutation tests, stability
├── backtests/
│   └── engine.py               # Shared backtest framework
├── data/
│   ├── raw/                    # Downloaded data (not in repo)
│   ├── processed/              # Analyzed data
│   └── features/               # ML feature matrices
└── output/
    ├── figures/                # 50+ generated plots (dashboards, backtests, stability)
    └── reports/                # Strategy comparison, ranking, robustness CSVs
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline (downloads data + runs all 12 analyses)
python run_all.py

# Run with cached data (skip downloads)
python run_all.py --skip-collect

# Run specific plans only
python run_all.py --skip-collect --plan 1 3 11 12

# Run advanced strategy research (30 strategies)
python analysis/advanced_research.py

# Run robustness validation (bootstrap, permutation tests)
python analysis/robustness.py
```

### Optional API Keys (for enhanced data)
```bash
export FINNHUB_API_KEY="your_key"        # Free at finnhub.io
export ALPHA_VANTAGE_KEY="your_key"      # Free at alphavantage.co
export REDDIT_CLIENT_ID="your_id"        # For PRAW (enhanced Reddit)
export REDDIT_SECRET="your_secret"
```

## Generated Outputs

### Figures (50+ plots)
- `onds_technical_dashboard.png` — Price + RSI + MACD + Volume
- `sector_comparison.png` — ONDS vs drone/defense peers (normalized)
- `crossasset_correlation_heatmap.png` — Full correlation matrix
- `onds_iv_smile.png` / `onds_iv_surface.png` — IV smile and 3D surface
- `regime_hmm.png` / `regime_volatility.png` / `regime_markovswitching.png` — Regime overlays
- `sentiment_reddit_vs_price.png` / `sentiment_news_vs_price.png`
- `feature_importance.png` — Top ML features
- `strategy_comparison.png` — All strategies overlaid
- `rolling_sharpe_stability.png` — Rolling 60-day Sharpe for top strategies
- 30x backtest equity curves (`bt_*.png`)

### Reports
- `strategy_comparison.csv` — All 30 strategies side-by-side
- `strategy_ranking.csv` — Strategies ranked by Sharpe ratio
- `advanced_strategy_comparison.csv` — Detailed advanced research comparison
- `robustness_report.csv` — Bootstrap CI, permutation p-values

### Data Files
- `technical_predictive_power.csv` — Spearman correlations for each indicator
- `crossasset_predictive_power.csv` — Cross-asset predictive tests
- `sector_lead_lag.csv` — Peer lead-lag analysis
- `event_study_results.csv` — CAR for 12 ONDS events
- `reddit_daily_sentiment.csv` / `news_daily_sentiment.csv`
- `regimes_hmm.csv` / `regimes_markov.csv` / `regimes_volatility.csv`
- `feature_importance.csv` — ML feature rankings
- `advanced_features.csv` — 106-feature matrix for ML experiments

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

### Advanced Strategy Types
- **Momentum+Stop**: SMA10/SMA50 trend following with 10% trailing stop and 15% max loss
- **Market Contrarian**: Fade NASDAQ/SPY extremes (z > 0.75) based on negative next-day correlation with ONDS
- **Adaptive Ensemble**: Re-weight 7 base signals every 20 days by trailing 40-day Sharpe (exponential weighting)
- **ML Direction**: Walk-forward Random Forest using top-15 mutual-information-selected features
- **DIX Enhanced**: Rolling percentile of dark pool DIX with GEX amplifier
- **Multi-TF Momentum**: Align 5d/10d/20d momentum — only trade when all agree
- **Volatility-adjusted**: Scale positions inversely to 20-day realized vol targeting 30% annualized portfolio vol

### Robustness Testing
- **Block bootstrap** (5000 samples, block_size=5): Preserves autocorrelation in daily returns
- **Permutation test** (2000 shuffles): Computes p-value for strategy Sharpe vs random signals
- **Rolling Sharpe**: 60-day window to check temporal stability
- **Drawdown analysis**: Duration, recovery time, worst drawdown periods

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
