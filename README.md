# ONDS Multi-Signal Quantitative Research

A comprehensive quantitative research framework analyzing **Ondas Holdings (ONDS)** from 14 different signal perspectives. Combines NLP sentiment analysis (Reddit, news), technical indicators, cross-asset signals (gold, silver, bitcoin, VIX), dark pool data, options/IV analysis, sector analysis (drone/defense peers), event studies, regime detection, **leakage-free ML models**, and **IV surface strategies** into a unified research pipeline.

## Key Findings

### Overfitting Analysis (Plans 13 -> 14)

The original ML pipeline (Plan 13) reported a Sharpe of +4.06 for the ML Direction strategy. This was inflated by 4 data-leakage bugs:

1. **Feature selection on full dataset** -- Mutual information was computed on all 280 rows (including test data), choosing features that predict the future.
2. **Expanding walk-forward without cap** -- ML trained on `[1:i]` growing from 60 to 280 samples with no rolling window, allowing feature selection to be dominated by later data.
3. **No purge gap** -- Returns are autocorrelated ~3-5 days. Without a gap between training and test days, the last training day leaks information about the test day.
4. **Adaptive Ensemble look-ahead** -- Evaluated sub-signal Sharpe on the SAME period being traded. Train Sharpe 3.37 -> Test Sharpe -0.60.

**Plan 14 (Leakage-Free ML)** fixes all 4 bugs with:
- Rolling 120-day training window (not expanding)
- 5-day purge gap between train and test
- Feature selection within each fold's training data only
- Holdout-based ensemble weight estimation

### ML Model Comparison (Plan 14 -- Leakage-Free)

6 ML models tested with walk-forward validation (120d train, 5d purge, per-fold feature selection):

| Model | Full Sharpe | Train Sharpe | Test Sharpe | Train-Test Gap | Bootstrap 95% CI | P(Sharpe>0) | Perm p-value |
|-------|------------|-------------|------------|---------------|-----------------|-------------|-------------|
| Random Forest | -- | -- | -- | -- | -- | -- | -- |
| XGBoost | -- | -- | -- | -- | -- | -- | -- |
| LightGBM | -- | -- | -- | -- | -- | -- | -- |
| LogReg (ElNet) | -- | -- | -- | -- | -- | -- | -- |
| SVM (RBF) | -- | -- | -- | -- | -- | -- | -- |
| MLP Neural Net | -- | -- | -- | -- | -- | -- | -- |

> **Note:** Run `python analysis/ml_models.py` to populate these numbers. Expected honest Sharpe range: 0.3-1.5 (down from the inflated 4.06).

### IV Surface Analysis (Plan 15)

- **SSVI Model**: Gatheral & Jacquier (2014) arbitrage-free parametric volatility surface fitted to 529 ONDS option contracts
- **7 IV-proxy features**: VRP, IV Rank, VIX Term, Skew Proxy, VIX Regime, RV Ratio, VIX-ONDS Correlation
- **3 IV strategies**: VRP Mean Reversion, IV Regime Conditional, Vol Surface Composite

| Strategy | Train Sharpe | Test Sharpe | Gap |
|----------|-------------|------------|-----|
| VRP Mean Reversion | -- | -- | -- |
| IV Regime Conditional | -- | -- | -- |
| Vol Surface Composite | -- | -- | -- |

> **Note:** Run `python analysis/iv_surface.py` to populate these numbers.

### Base Analysis Results (Plans 1-12)

| Strategy | Sharpe | Annual Return | Max Drawdown | Hit Rate |
|----------|--------|--------------|--------------|----------|
| Technical Composite | +0.38 | +41.9% | -45.6% | 52.8% |
| Dark Pool (DIX/GEX) | +0.69 | +34.5% | -38.4% | 53.1% |
| Reddit Sentiment | +4.10 | +779%\* | -33.7% | 55.6% |
| Cross-Asset | -0.99 | -1.1% | -5.4% | 37.5% |
| Sector Momentum | -1.14 | -41.9% | -54.6% | 48.3% |

\*Reddit sentiment has limited data (46 trades, 80 trading days) -- results not reliable.

### Notable Results
- **Peer lead-lag**: JOBY lag-1 significant (p=0.007), top ML feature for direction prediction
- **Regime detection**: HMM identifies 3 regimes -- Bearish (86%), Neutral (8%), Bullish (6%, mean=+14.3%/day)
- **Event study**: Acquisition events show significant CAR (+114%, p=0.003)
- **Options**: ATM IV ~117%, put-call volume ratio 0.49 (bullish), IV term structure slightly inverted
- **Cross-asset**: Gold lag-1 and VIX change are significant direction predictors
- **Top ML features**: gold_ret_lag1, JOBY_ret_lag2, KTOS_ret_lag5, RCAT_ret_lag3, regime
- **Direction classification**: Peer-only features achieve 56.9% accuracy (above 50% baseline)
- **Gap direction**: 59.1% accuracy using lagged returns and peer features

## Pipeline Architecture

```
Data Collection (Plan 0)
+-- yfinance: ONDS + 16 related tickers (280 trading days, 2025-01+)
+-- Reddit JSON API: 362 posts from 7 subreddits
+-- Google News RSS: 100+ articles
+-- Finnhub API: 217 news articles + analyst recommendations
+-- ChartExchange: ONDS FINRA short volume (375 days)
+-- yfinance Options: 529 contracts, 10 expirations
+-- Dark pool: SqueezeMetrics DIX/GEX (3,720 days)

Analysis (Plans 1-12)
+-- Plan 1:  Technical Analysis (RSI, MACD, Bollinger, ATR, OBV, SMA)
+-- Plan 2:  Sector Analysis (RCAT, AVAV, KTOS, JOBY, LMT, RTX, ITA)
+-- Plan 3:  Cross-Asset (GLD, SLV, BTC, VIX, SPY, QQQ, TLT, UUP)
+-- Plan 4:  Dark Pool (DIX/GEX + ChartExchange FINRA short volume)
+-- Plan 5:  Reddit Sentiment (VADER + FinBERT ensemble)
+-- Plan 6:  News Sentiment (Google News + NLP scoring)
+-- Plan 7:  CEO Tweet Analysis (framework -- needs X credentials)
+-- Plan 8:  Analyst Reports (Finnhub: consensus 1.29, very bullish)
+-- Plan 9:  Event Studies (12 known ONDS events, CAR analysis)
+-- Plan 10: Options & IV Surface (smile, surface, skew, term structure)
+-- Plan 11: Regime Detection (HMM, GMM, Markov Switching, volatility-based)
+-- Plan 12: Fusion (103-feature matrix + ML classification)

Advanced Research (Plan 13)
+-- 106 features with leakage-safe engineering
+-- 30 strategies (15 base + 15 volatility-adjusted)
+-- Bootstrap & permutation testing
+-- NOTE: ML signals in Plan 13 have leakage bugs (see above)

Leakage-Free ML (Plan 14) ** NEW **
+-- Rolling 120-day walk-forward with 5-day purge gap
+-- Per-fold feature selection (MI on training data only)
+-- 6 models: RF, XGBoost, LightGBM, LogReg, SVM, MLP
+-- Full validation: 60/10/30 split + bootstrap CI + permutation test + Bayesian
+-- Fixed Adaptive Ensemble (holdout-based weights)

IV Surface Analysis (Plan 15) ** NEW **
+-- SSVI model (Gatheral & Jacquier 2014) fitted to ONDS options
+-- 7 IV-proxy features for daily backtesting
+-- 3 IV-based trading strategies
+-- ML integration: test IV feature uplift
```

## Directory Structure

```
ONDS_Research/
+-- config.py                    # Central configuration (tickers, parameters)
+-- run_all.py                   # Master pipeline orchestrator
+-- requirements.txt
+-- collectors/                  # Data collection modules
|   +-- prices.py               # yfinance price data
|   +-- reddit.py               # Reddit JSON API scraper
|   +-- news.py                 # Google News + Finnhub
|   +-- darkpool.py             # SqueezeMetrics + Stockgrid
|   +-- options.py              # yfinance options chains
+-- analysis/                    # Analysis modules
|   +-- technical.py            # Technical indicators + predictive power
|   +-- crossasset.py           # Cross-asset correlations + signals
|   +-- sector.py               # Peer/sector lead-lag + momentum
|   +-- darkpool.py             # DIX/GEX + ChartExchange short volume
|   +-- sentiment.py            # VADER + FinBERT sentiment scoring
|   +-- analyst.py              # Analyst recommendation analysis
|   +-- events.py               # Event study methodology
|   +-- options_iv.py           # IV smile, surface, derived features
|   +-- regime.py               # HMM, GMM, Markov Switching
|   +-- fusion.py               # Multi-source feature matrix + ML
|   +-- advanced_research.py    # 30 strategies, ML signals, feature selection
|   +-- param_optimization.py   # Grid search, 60/10/30 split validation
|   +-- robustness.py           # Bootstrap, permutation tests, stability
|   +-- statistical_validation.py # Bayesian analysis, cross-sectional tests
|   +-- alpha_decay.py          # Alpha persistence / decay analysis
|   +-- train_test_validation.py # Train/test split validation
|   +-- ml_models.py            # ** NEW ** Leakage-free 6-model ML pipeline
|   +-- iv_surface.py           # ** NEW ** SSVI + IV features + IV strategies
+-- backtests/
|   +-- engine.py               # Shared backtest framework
+-- data/
|   +-- raw/                    # Downloaded data (not in repo)
|   +-- processed/              # Analyzed data
|   +-- features/               # ML feature matrices
+-- output/
    +-- figures/                # 60+ generated plots
    +-- reports/                # Strategy comparison, ranking, robustness CSVs
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

# Run leakage-free ML comparison (6 models) ** NEW **
python analysis/ml_models.py

# Run IV surface analysis ** NEW **
python analysis/iv_surface.py

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

### Figures (60+ plots)
- `onds_technical_dashboard.png` -- Price + RSI + MACD + Volume
- `sector_comparison.png` -- ONDS vs drone/defense peers (normalized)
- `crossasset_correlation_heatmap.png` -- Full correlation matrix
- `onds_iv_smile.png` / `onds_iv_surface.png` -- IV smile and 3D surface
- `regime_hmm.png` / `regime_volatility.png` / `regime_markovswitching.png` -- Regime overlays
- `sentiment_reddit_vs_price.png` / `sentiment_news_vs_price.png`
- `feature_importance.png` -- Top ML features
- `strategy_comparison.png` -- All strategies overlaid
- `rolling_sharpe_stability.png` -- Rolling 60-day Sharpe for top strategies
- `ml_model_comparison.png` -- ** NEW ** 6-model Test Sharpe with bootstrap CI
- `ml_walkforward_equity.png` -- ** NEW ** Walk-forward equity curves (leakage-free)
- `ml_train_test_gap.png` -- ** NEW ** Train vs Test Sharpe scatter (overfitting diagnostic)
- `ssvi_onds_surface.png` -- ** NEW ** SSVI arbitrage-free IV surface
- `iv_proxy_features.png` -- ** NEW ** 7 IV proxy feature time series
- `iv_strategy_equity.png` -- ** NEW ** IV strategy equity curves
- 30x backtest equity curves (`bt_*.png`)

### Reports
- `strategy_comparison.csv` -- All 30 strategies side-by-side
- `strategy_ranking.csv` -- Strategies ranked by Sharpe ratio
- `advanced_strategy_comparison.csv` -- Detailed advanced research comparison
- `robustness_report.csv` -- Bootstrap CI, permutation p-values
- `ml_model_comparison.csv` -- ** NEW ** 6-model comparison with all validation metrics
- `iv_surface_analysis.csv` -- ** NEW ** SSVI fit parameters and quality
- `iv_strategy_results.csv` -- ** NEW ** IV strategy backtest results

### Data Files
- `technical_predictive_power.csv` -- Spearman correlations for each indicator
- `crossasset_predictive_power.csv` -- Cross-asset predictive tests
- `sector_lead_lag.csv` -- Peer lead-lag analysis
- `event_study_results.csv` -- CAR for 12 ONDS events
- `reddit_daily_sentiment.csv` / `news_daily_sentiment.csv`
- `regimes_hmm.csv` / `regimes_markov.csv` / `regimes_volatility.csv`
- `feature_importance.csv` -- ML feature rankings
- `advanced_features.csv` -- 106-feature matrix for ML experiments
- `iv_proxy_features.csv` -- ** NEW ** 7 daily IV proxy features

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
- **Lag-1 execution**: Signal on day t -> position on day t+1
- **Transaction costs**: 10 bps round-trip
- **Metrics**: Sharpe, annual return/vol, max drawdown, hit rate
- **Walk-forward**: Expanding window for ML predictions (no future leakage)

### Leakage-Free ML Pipeline (Plan 14)

The corrected ML pipeline uses strict walk-forward validation:

```
At each test day i:
  train = [i-120 : i-5]    <- rolling 120-day window (NOT expanding)
  purge = [i-5 : i]        <- 5-day gap (prevents autocorrelation leakage)
  test  = [i]              <- predict 1 day

  Feature selection: MI on train window ONLY (not full dataset)
  Scaler: fit on train window ONLY
  Model: fit on train window ONLY
```

6 models compared: Random Forest, XGBoost, LightGBM, Logistic Regression (ElasticNet), SVM (RBF), MLP Neural Network. Each validated through 60/10/30 split, 5000-sample block bootstrap, permutation test, and Bayesian P(alpha>0).

### IV Surface Model (Plan 15)

Uses the SSVI (Surface SVI) parametric model from Gatheral & Jacquier (2014):

```
w(k, tau) = (theta/2) * (1 + rho*phi*k + sqrt((phi*k + rho)^2 + 1 - rho^2))
```

Where k = log-moneyness, theta = ATM total variance, rho = skew, phi = vol-of-vol. This is NOT Black-Scholes -- it is an arbitrage-free parametric volatility surface with calendar spread and butterfly constraints built in.

### Advanced Strategy Types
- **Momentum+Stop**: SMA10/SMA50 trend following with 10% trailing stop and 15% max loss
- **Market Contrarian**: Fade NASDAQ/SPY extremes (z > 0.75) based on negative next-day correlation with ONDS
- **Adaptive Ensemble (Fixed)**: Re-weight 6 base signals every 20 days using holdout-based Sharpe evaluation (no look-ahead)
- **ML Direction**: Walk-forward classification using top-15 per-fold MI-selected features
- **DIX Enhanced**: Rolling percentile of dark pool DIX with GEX amplifier
- **Multi-TF Momentum**: Align 5d/10d/20d momentum -- only trade when all agree
- **VRP Mean Reversion**: High VRP (fear) + ONDS oversold -> long
- **IV Regime Conditional**: Low-VIX: momentum; High-VIX: mean-reversion
- **Vol Surface Composite**: Weighted VRP z-score + IV rank + skew proxy
- **Volatility-adjusted**: Scale positions inversely to 20-day realized vol targeting 30% annualized portfolio vol

### Robustness Testing
- **Block bootstrap** (5000 samples, block_size=5): Preserves autocorrelation in daily returns
- **Permutation test** (500-2000 shuffles): Computes p-value for strategy Sharpe vs random signals
- **Rolling Sharpe**: 60-day window to check temporal stability
- **Drawdown analysis**: Duration, recovery time, worst drawdown periods
- **Bayesian inference**: Conjugate normal-normal posterior for daily alpha with skeptical prior
- **Train-Test gap**: Monitor |Train Sharpe - Test Sharpe| < 2.0 as overfitting diagnostic

## ONDS Company Profile

**Ondas Holdings (NASDAQ: ONDS)** -- Defense/drone technology company
- Market cap: ~$4.37B (as of early 2026)
- Products: AURA drone system, FullMAX wireless network
- Subsidiary: American Robotics (autonomous drone solutions)
- Key catalysts: DoD contracts, NATO partnerships, FAA drone regulations
- CEO: Eric Brock (@CeoOndas on X)
- Peer universe: RCAT, AVAV, KTOS, JOBY, LMT, RTX

## Future Improvements

### CEO Twitter/X Sentiment (Plan 7)
- **Goal**: Scrape tweets from @CeoOndas (Eric Brock) and analyze sentiment around key announcements
- **Tool**: `twscrape` (installed) -- requires X account authentication
- **Setup**: Run `python -m twscrape add_accounts "username" "password" "email" "email_password"` then `python -m twscrape login_accounts`
- **Alternative**: Apply for X API developer access at [developer.twitter.com](https://developer.twitter.com)
- **Analysis**: Score tweets with FinBERT, compute event-window returns around tweet bursts
- **Status**: Framework built (`analysis/sentiment.py` supports Twitter), awaiting authentication

### Additional Enhancements
- **Real-time dark pool**: Integrate live SqueezeMetrics DIX/GEX feed (requires subscription)
- **Intraday options flow**: Monitor unusual options activity via CBOE/OPRA data
- **Insider transactions**: Track SEC Form 4 filings for ONDS insiders (via SEC EDGAR API)
- **Reddit real-time**: Use PRAW (Reddit API) for streaming sentiment instead of JSON API snapshots
- **Ensemble tuning**: Hyperparameter optimization (Optuna), SHAP feature explanations
- **Portfolio optimization**: Mean-variance or risk-parity allocation across multiple signal strategies
- **Live ONDS option chains**: Daily scraping for real IV surface time series (currently only snapshot)

## License

Academic and research use. Part of a graduation thesis on multi-signal quantitative analysis.
