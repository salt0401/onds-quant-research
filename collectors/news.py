"""
Plan 6: News Data Collection for ONDS

Collects news articles from free APIs:
1. Finnhub (free tier: 60 calls/min)
2. Alpha Vantage News Sentiment API
3. Google News RSS feed (no API key needed)
"""
import requests
import pandas as pd
import time
import xml.etree.ElementTree as ET
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import TARGET_TICKER, FINNHUB_API_KEY, ALPHA_VANTAGE_KEY, DATA_RAW


def collect_google_news(query: str = "ONDS stock Ondas Holdings", save: bool = True) -> pd.DataFrame:
    """
    Collect news from Google News RSS (no API key needed).
    Returns headlines, dates, and source URLs.
    """
    print(f"  Collecting Google News for '{query}'...")

    url = "https://news.google.com/rss/search"
    params = {
        "q": query,
        "hl": "en-US",
        "gl": "US",
        "ceid": "US:en",
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        if resp.status_code != 200:
            print(f"    HTTP {resp.status_code}")
            return pd.DataFrame()

        root = ET.fromstring(resp.content)
        items = root.findall(".//item")

        articles = []
        for item in items:
            articles.append({
                "title":     item.findtext("title", ""),
                "link":      item.findtext("link", ""),
                "pubDate":   item.findtext("pubDate", ""),
                "source":    item.findtext("source", ""),
            })

        df = pd.DataFrame(articles)
        if not df.empty and "pubDate" in df.columns:
            df["datetime"] = pd.to_datetime(df["pubDate"], errors="coerce")
            df["date"] = df["datetime"].dt.date
            df = df.sort_values("datetime", ascending=False)

        print(f"    Collected {len(df)} news articles from Google News")

        if save and not df.empty:
            path = DATA_RAW / "google_news_onds.csv"
            df.to_csv(path, index=False)
            print(f"    Saved: {path}")

        return df

    except Exception as e:
        print(f"    Failed: {e}")
        return pd.DataFrame()


def collect_finnhub_news(ticker: str = TARGET_TICKER, save: bool = True) -> pd.DataFrame:
    """
    Collect company news from Finnhub API.
    Free tier: 60 API calls/minute.
    """
    if not FINNHUB_API_KEY:
        print("  No FINNHUB_API_KEY set. Skipping Finnhub news.")
        print("  Get free key at: https://finnhub.io/register")
        return pd.DataFrame()

    print(f"  Collecting Finnhub news for {ticker}...")

    # Collect in chunks (Finnhub limits date range)
    from datetime import datetime, timedelta
    all_articles = []
    end = datetime.now()
    start = end - timedelta(days=365)

    url = "https://finnhub.io/api/v1/company-news"
    params = {
        "symbol": ticker,
        "from": start.strftime("%Y-%m-%d"),
        "to": end.strftime("%Y-%m-%d"),
        "token": FINNHUB_API_KEY,
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 200:
            articles = resp.json()
            if isinstance(articles, list):
                all_articles.extend(articles)
                print(f"    Got {len(articles)} articles from Finnhub")
    except Exception as e:
        print(f"    Finnhub failed: {e}")

    if all_articles:
        df = pd.DataFrame(all_articles)
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], unit="s")
            df["date"] = df["datetime"].dt.date
        if save:
            path = DATA_RAW / f"finnhub_news_{ticker}.csv"
            df.to_csv(path, index=False)
            print(f"    Saved: {path}")
        return df

    return pd.DataFrame()


def collect_finnhub_recommendations(ticker: str = TARGET_TICKER, save: bool = True) -> pd.DataFrame:
    """
    Collect analyst recommendations from Finnhub (buy/hold/sell).
    This is Plan 8 data but collected here since it's the same API.
    """
    if not FINNHUB_API_KEY:
        print("  No FINNHUB_API_KEY. Skipping analyst recommendations.")
        return pd.DataFrame()

    print(f"  Collecting Finnhub analyst recommendations for {ticker}...")

    url = "https://finnhub.io/api/v1/stock/recommendation"
    params = {"symbol": ticker, "token": FINNHUB_API_KEY}

    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            df = pd.DataFrame(data)
            if not df.empty:
                df["period"] = pd.to_datetime(df["period"])
                df = df.sort_values("period")
                print(f"    Got {len(df)} recommendation periods")
                if save:
                    path = DATA_RAW / f"finnhub_recommendations_{ticker}.csv"
                    df.to_csv(path, index=False)
                return df
    except Exception as e:
        print(f"    Failed: {e}")

    return pd.DataFrame()


def collect_finnhub_price_target(ticker: str = TARGET_TICKER, save: bool = True) -> pd.DataFrame:
    """Collect analyst price targets from Finnhub."""
    if not FINNHUB_API_KEY:
        return pd.DataFrame()

    print(f"  Collecting Finnhub price targets for {ticker}...")
    url = "https://finnhub.io/api/v1/stock/price-target"
    params = {"symbol": ticker, "token": FINNHUB_API_KEY}

    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            if data:
                df = pd.DataFrame([data])
                if save:
                    path = DATA_RAW / f"finnhub_price_target_{ticker}.csv"
                    df.to_csv(path, index=False)
                print(f"    Price target data: {data}")
                return df
    except Exception as e:
        print(f"    Failed: {e}")

    return pd.DataFrame()


if __name__ == "__main__":
    print("=" * 60)
    print("NEWS & ANALYST DATA COLLECTION")
    print("=" * 60)
    gn = collect_google_news()
    fn = collect_finnhub_news()
    rec = collect_finnhub_recommendations()
    pt = collect_finnhub_price_target()
    print("\nDone.")
