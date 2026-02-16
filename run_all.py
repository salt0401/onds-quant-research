"""
ONDS Quantitative Research — Master Pipeline

Runs all 12 research plans in sequence:
1. Collect all data (prices, reddit, news, dark pool, options)
2. Analyze each signal source
3. Backtest each strategy
4. Compare all strategies
5. Run regime detection
6. Build fusion model

Usage:
    python run_all.py                    # Run everything
    python run_all.py --skip-collect     # Skip data collection (use cached)
    python run_all.py --plan 1 3 4       # Run only specific plans
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from config import TARGET_TICKER, DATA_RAW, DATA_PROC, FIGURES_DIR


def banner(text: str):
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}")


def section(text: str):
    print(f"\n--- {text} ---")


def run_plan_0(skip_collect: bool = False):
    """Plan 0: Data Collection — download all raw data."""
    banner("PLAN 0: DATA COLLECTION")

    if skip_collect:
        print("  Skipping data collection (--skip-collect)")
        # Verify we have price data at minimum
        if not (DATA_RAW / "ONDS_prices.csv").exists():
            print("  ERROR: No cached price data. Run without --skip-collect first.")
            sys.exit(1)
        return

    # 0a. Price data (yfinance — no API key needed)
    section("0a. Downloading price data (yfinance)")
    from collectors.prices import download_all_prices
    price_data = download_all_prices()
    print(f"  Downloaded {len(price_data)} tickers")

    # 0b. Reddit
    section("0b. Collecting Reddit posts")
    try:
        from collectors.reddit import collect_from_multiple_subreddits
        reddit_df = collect_from_multiple_subreddits()
        print(f"  Reddit: {len(reddit_df)} posts")
    except Exception as e:
        print(f"  Reddit collection failed: {e}")

    # 0c. News
    section("0c. Collecting news articles")
    try:
        from collectors.news import (collect_google_news, collect_finnhub_news,
                                      collect_finnhub_recommendations, collect_finnhub_price_target)
        collect_google_news()
        collect_finnhub_news()
        collect_finnhub_recommendations()
        collect_finnhub_price_target()
    except Exception as e:
        print(f"  News collection failed: {e}")

    # 0d. Dark pool
    section("0d. Collecting dark pool data")
    try:
        from collectors.darkpool import (collect_squeezemetrics_dix,
                                          collect_stockgrid_darkpool, collect_finra_short_volume)
        collect_squeezemetrics_dix()
        collect_stockgrid_darkpool()
        collect_finra_short_volume()
    except Exception as e:
        print(f"  Dark pool collection failed: {e}")

    # 0e. Options
    section("0e. Collecting options data")
    try:
        from collectors.options import collect_options_chain
        collect_options_chain()
    except Exception as e:
        print(f"  Options collection failed: {e}")


def run_plan_1():
    """Plan 1: ONDS Price & Technical Analysis."""
    banner("PLAN 1: TECHNICAL ANALYSIS")
    from collectors.prices import load_prices
    from analysis.technical import compute_indicators, generate_signals, analyze_predictive_power, plot_technical_dashboard
    from backtests.engine import backtest, print_results

    prices = load_prices(TARGET_TICKER)
    df = compute_indicators(prices)
    df.to_csv(DATA_PROC / "onds_technical.csv")

    section("1a. Predictive power of technical indicators")
    analyze_predictive_power(df)

    section("1b. Technical dashboard")
    plot_technical_dashboard(df)

    section("1c. Backtest technical signals")
    signals = generate_signals(df)
    results = backtest(prices["Close"], signals["signal"], name="Technical Composite")
    print_results(results)

    # Individual indicator backtests
    for sig_name in ["rsi_sig", "macd_sig", "sma_sig"]:
        if sig_name in signals.columns:
            r = backtest(prices["Close"], signals[sig_name], name=f"Technical_{sig_name}",
                        plot=False)
            print_results(r)

    return results


def run_plan_2():
    """Plan 2: Related Stocks & Sector Analysis."""
    banner("PLAN 2: SECTOR ANALYSIS")
    from collectors.prices import load_all_closes
    from analysis.sector import compute_sector_features, test_lead_lag, generate_sector_signal, plot_sector_comparison
    from backtests.engine import backtest, print_results
    from collectors.prices import load_prices

    closes = load_all_closes()
    prices = load_prices(TARGET_TICKER)

    section("2a. Sector comparison plot")
    plot_sector_comparison(closes)

    section("2b. Lead-lag analysis")
    test_lead_lag(closes)

    section("2c. Sector features & backtest")
    features = compute_sector_features(closes)
    features.to_csv(DATA_PROC / "sector_features.csv")
    signal = generate_sector_signal(features)
    results = backtest(prices["Close"], signal, name="Sector Momentum")
    print_results(results)

    return results


def run_plan_3():
    """Plan 3: Cross-Asset Signals."""
    banner("PLAN 3: CROSS-ASSET SIGNALS")
    from collectors.prices import load_all_closes, load_prices
    from analysis.crossasset import (compute_cross_asset_features, test_predictive_power,
                                      generate_cross_asset_signal, plot_correlation_heatmap)
    from backtests.engine import backtest, print_results

    closes = load_all_closes()
    prices = load_prices(TARGET_TICKER)

    section("3a. Correlation heatmap")
    plot_correlation_heatmap(closes)

    section("3b. Cross-asset features")
    features = compute_cross_asset_features(closes)
    features.to_csv(DATA_PROC / "crossasset_features.csv")

    section("3c. Predictive power tests")
    test_predictive_power(features)

    section("3d. Backtest cross-asset signal")
    signal = generate_cross_asset_signal(features)
    results = backtest(prices["Close"], signal, name="Cross-Asset")
    print_results(results)

    return results


def run_plan_4():
    """Plan 4: Dark Pool Analysis."""
    banner("PLAN 4: DARK POOL ANALYSIS")
    from collectors.prices import load_prices
    from analysis.darkpool import (load_dix_data, analyze_dix_predictive_power,
                                    generate_darkpool_signal, plot_dix_analysis)
    from backtests.engine import backtest, print_results

    prices = load_prices(TARGET_TICKER)
    dix_df = load_dix_data()

    if not dix_df.empty:
        section("4a. DIX predictive power")
        analyze_dix_predictive_power(dix_df, prices)

        section("4b. DIX visualization")
        plot_dix_analysis(dix_df, prices)

        section("4c. Backtest dark pool signal")
        signal = generate_darkpool_signal(dix_df)
        results = backtest(prices["Close"], signal, name="Dark Pool (DIX/GEX)")
        print_results(results)
        return results
    else:
        print("  No dark pool data available. Skipping backtest.")
        return None


def run_plan_5():
    """Plan 5: Reddit Sentiment Analysis."""
    banner("PLAN 5: REDDIT SENTIMENT")
    from collectors.prices import load_prices
    from analysis.sentiment import (analyze_reddit_sentiment, test_sentiment_predictive_power,
                                     generate_sentiment_signal, plot_sentiment_vs_price)
    from backtests.engine import backtest, print_results

    prices = load_prices(TARGET_TICKER)

    section("5a. Score Reddit posts")
    daily = analyze_reddit_sentiment()

    if not daily.empty:
        section("5b. Predictive power")
        test_sentiment_predictive_power(daily, prices, source_name="Reddit")

        section("5c. Visualization")
        plot_sentiment_vs_price(daily, prices, source_name="Reddit")

        section("5d. Backtest Reddit sentiment signal")
        signal = generate_sentiment_signal(daily)
        results = backtest(prices["Close"], signal, name="Reddit Sentiment")
        print_results(results)
        return results
    else:
        print("  No Reddit data. Skipping.")
        return None


def run_plan_6():
    """Plan 6: News Sentiment Analysis."""
    banner("PLAN 6: NEWS SENTIMENT")
    from collectors.prices import load_prices
    from analysis.sentiment import (analyze_news_sentiment, test_sentiment_predictive_power,
                                     generate_sentiment_signal, plot_sentiment_vs_price)
    from backtests.engine import backtest, print_results

    prices = load_prices(TARGET_TICKER)

    section("6a. Score news articles")
    daily = analyze_news_sentiment()

    if not daily.empty:
        section("6b. Predictive power")
        test_sentiment_predictive_power(daily, prices, source_name="News")

        section("6c. Visualization")
        plot_sentiment_vs_price(daily, prices, source_name="News")

        section("6d. Backtest news sentiment signal")
        signal = generate_sentiment_signal(daily)
        results = backtest(prices["Close"], signal, name="News Sentiment")
        print_results(results)
        return results
    else:
        print("  No news data. Skipping.")
        return None


def run_plan_7():
    """Plan 7: CEO Tweet Analysis (placeholder — needs Twitter data)."""
    banner("PLAN 7: CEO TWEET ANALYSIS")
    print("  Twitter/X scraping requires authentication.")
    print("  To collect data: set up twscrape with X credentials.")
    print("  Target account: @CeoOndas")
    print("  Skipping for now — data collection code in collectors/tweets.py")
    return None


def run_plan_8():
    """Plan 8: Analyst Reports & Target Prices."""
    banner("PLAN 8: ANALYST REPORTS")
    from collectors.prices import load_prices
    from analysis.analyst import load_recommendations, analyze_recommendations, plot_analyst_history
    from backtests.engine import backtest, print_results

    prices = load_prices(TARGET_TICKER)
    rec_df = load_recommendations()

    if not rec_df.empty:
        section("8a. Analyst recommendation analysis")
        analyze_recommendations(rec_df, prices)

        section("8b. Visualization")
        plot_analyst_history(rec_df, prices)
        return None
    else:
        print("  No analyst data. Set FINNHUB_API_KEY to collect.")
        return None


def run_plan_9():
    """Plan 9: Government Contracts & Event Studies."""
    banner("PLAN 9: EVENT STUDY ANALYSIS")
    from collectors.prices import load_prices
    from analysis.events import run_all_event_studies, KNOWN_EVENTS

    prices = load_prices(TARGET_TICKER)

    section("9a. Event studies for known ONDS events")
    results = run_all_event_studies(prices)
    return None


def run_plan_10():
    """Plan 10: Options & IV Surface Analysis."""
    banner("PLAN 10: OPTIONS & IV ANALYSIS")
    from analysis.options_iv import load_options_data, compute_iv_summary, plot_iv_smile, plot_iv_surface
    import yfinance as yf

    options_df = load_options_data()
    if not options_df.empty:
        stock = yf.Ticker(TARGET_TICKER)
        hist = stock.history(period="5d")
        current_price = hist["Close"].iloc[-1] if not hist.empty else 0

        section("10a. IV summary")
        summary = compute_iv_summary(options_df, current_price)
        for k, v in summary.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            else:
                print(f"    {k}: {v}")

        section("10b. IV smile plot")
        plot_iv_smile(options_df, current_price)

        section("10c. IV surface plot")
        plot_iv_surface(options_df, current_price)
    else:
        print("  No options data. Run collectors/options.py first.")
    return None


def run_plan_11():
    """Plan 11: Regime Detection & Switching Models."""
    banner("PLAN 11: REGIME DETECTION")
    from collectors.prices import load_prices
    from analysis.regime import (detect_regimes_hmm, detect_regimes_gmm,
                                  detect_regimes_volatility, detect_regimes_markov_switching,
                                  plot_regimes)

    prices = load_prices(TARGET_TICKER)
    returns = prices["Close"].pct_change().dropna()

    section("11a. HMM regime detection")
    hmm_df = detect_regimes_hmm(returns)
    if not hmm_df.empty:
        hmm_df.to_csv(DATA_PROC / "regimes_hmm.csv")
        plot_regimes(hmm_df, prices["Close"], method="HMM")

    section("11b. Volatility-based regimes")
    vol_df = detect_regimes_volatility(returns)
    vol_df.to_csv(DATA_PROC / "regimes_volatility.csv")
    plot_regimes(vol_df, prices["Close"], method="Volatility")

    section("11c. Markov Switching model")
    ms_df = detect_regimes_markov_switching(returns, n_regimes=2)
    if not ms_df.empty:
        ms_df.to_csv(DATA_PROC / "regimes_markov.csv")
        plot_regimes(ms_df, prices["Close"], method="MarkovSwitching")

    return hmm_df


def run_plan_12(all_results: dict):
    """Plan 12: Multi-Source Fusion."""
    banner("PLAN 12: MULTI-SOURCE FUSION")
    from collectors.prices import load_prices
    from analysis.fusion import build_feature_matrix, train_ensemble_model, generate_fusion_signal
    from backtests.engine import backtest, print_results, compare_strategies

    prices = load_prices(TARGET_TICKER)

    # Load all available processed data
    tech_df = None
    xa_feat = None
    sec_feat = None
    reddit_sent = None
    news_sent = None
    regime_df = None

    import pandas as pd
    sources = {
        "technical": ("onds_technical.csv", "Date"),
        "crossasset": ("crossasset_features.csv", "Date"),
        "sector": ("sector_features.csv", "Date"),
        "reddit": ("reddit_daily_sentiment.csv", "date"),
        "news": ("news_daily_sentiment.csv", "date"),
        "regime": ("regimes_hmm.csv", None),
    }
    loaded = {}
    for name, (fname, idx_col) in sources.items():
        fpath = DATA_PROC / fname
        try:
            kw = {"index_col": idx_col, "parse_dates": True} if idx_col else {"index_col": 0, "parse_dates": True}
            loaded[name] = pd.read_csv(fpath, **kw)
            print(f"    Loaded {name}: {loaded[name].shape}")
        except Exception as e:
            print(f"    FAILED {name}: {e}")
            loaded[name] = None
    tech_df = loaded["technical"]
    xa_feat = loaded["crossasset"]
    sec_feat = loaded["sector"]
    reddit_sent = loaded["reddit"]
    news_sent = loaded["news"]
    regime_df = loaded["regime"]

    section("12a. Build feature matrix")
    from config import DATA_FEAT
    import pandas as pd
    features = build_feature_matrix(
        onds_prices=prices,
        technical_df=tech_df,
        crossasset_features=xa_feat,
        sector_features=sec_feat,
        reddit_sentiment=reddit_sent,
        news_sentiment=news_sent,
        regime_df=regime_df,
    )
    features.to_csv(DATA_FEAT / "full_feature_matrix.csv")

    section("12b. Train ensemble model")
    model_results = train_ensemble_model(features)

    section("12c. Backtest fusion signal")
    fusion_signal = generate_fusion_signal(features)
    fusion_bt = backtest(prices["Close"], fusion_signal, name="Multi-Source Fusion")
    print_results(fusion_bt)

    # Compare all strategies
    section("12d. Strategy comparison")
    valid_results = [r for r in all_results.values() if r is not None and "cum_return" in r]
    if fusion_bt and "cum_return" in fusion_bt:
        valid_results.append(fusion_bt)
    if valid_results:
        compare_strategies(valid_results)


def main():
    parser = argparse.ArgumentParser(description="ONDS Quantitative Research Pipeline")
    parser.add_argument("--skip-collect", action="store_true", help="Skip data collection")
    parser.add_argument("--plan", nargs="+", type=int, help="Run specific plans (1-12)")
    args = parser.parse_args()

    start_time = time.time()

    banner("ONDS QUANTITATIVE RESEARCH PIPELINE")
    print(f"  Target: {TARGET_TICKER}")
    print(f"  Time:   {time.strftime('%Y-%m-%d %H:%M:%S')}")

    plans_to_run = set(args.plan) if args.plan else set(range(0, 13))

    # Plan 0: Data Collection
    if 0 in plans_to_run:
        run_plan_0(skip_collect=args.skip_collect)

    all_results = {}

    # Plans 1-11: Individual analyses
    plan_funcs = {
        1:  run_plan_1,
        2:  run_plan_2,
        3:  run_plan_3,
        4:  run_plan_4,
        5:  run_plan_5,
        6:  run_plan_6,
        7:  run_plan_7,
        8:  run_plan_8,
        9:  run_plan_9,
        10: run_plan_10,
        11: run_plan_11,
    }

    for plan_num in sorted(plans_to_run):
        if plan_num in plan_funcs:
            try:
                result = plan_funcs[plan_num]()
                all_results[f"plan_{plan_num}"] = result
            except Exception as e:
                print(f"\n  PLAN {plan_num} FAILED: {e}")
                import traceback
                traceback.print_exc()
                all_results[f"plan_{plan_num}"] = None

    # Plan 12: Fusion (depends on all others)
    if 12 in plans_to_run:
        try:
            run_plan_12(all_results)
        except Exception as e:
            print(f"\n  PLAN 12 (Fusion) FAILED: {e}")
            import traceback
            traceback.print_exc()

    elapsed = time.time() - start_time
    banner(f"PIPELINE COMPLETE — {elapsed:.0f}s elapsed")


if __name__ == "__main__":
    main()
