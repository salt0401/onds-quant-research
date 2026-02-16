"""
Plan 4: Dark Pool Data Collection

Scrapes:
1. SqueezeMetrics DIX/GEX (market-wide dark pool + gamma exposure)
2. Stockgrid dark pool data (per-ticker net short volume)
"""
import requests
import pandas as pd
import json
import time
from pathlib import Path
from bs4 import BeautifulSoup
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import TARGET_TICKER, DATA_RAW


def collect_squeezemetrics_dix(save: bool = True) -> pd.DataFrame:
    """
    Download DIX/GEX data from SqueezeMetrics.
    The site provides a downloadable CSV at their API endpoint.
    """
    print("  Collecting SqueezeMetrics DIX/GEX data...")

    # SqueezeMetrics provides data via their monitor page
    # Try the known data endpoint
    url = "https://squeezemetrics.com/monitor/dix"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    try:
        # Try to get the CSV download
        csv_url = "https://squeezemetrics.com/monitor/dix.csv"
        resp = requests.get(csv_url, headers=headers, timeout=30)
        if resp.status_code == 200 and "date" in resp.text.lower()[:100]:
            from io import StringIO
            df = pd.read_csv(StringIO(resp.text))
            df.columns = [c.strip() for c in df.columns]
            if "date" in [c.lower() for c in df.columns]:
                date_col = [c for c in df.columns if c.lower() == "date"][0]
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.set_index(date_col).sort_index()
            print(f"    Got {len(df)} rows from SqueezeMetrics CSV")
            if save:
                path = DATA_RAW / "squeezemetrics_dix_gex.csv"
                df.to_csv(path)
                print(f"    Saved: {path}")
            return df
    except Exception as e:
        print(f"    CSV download failed: {e}")

    # Fallback: try to scrape the page
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code == 200:
            # Look for embedded JSON data
            soup = BeautifulSoup(resp.text, "html.parser")
            scripts = soup.find_all("script")
            for script in scripts:
                text = script.string or ""
                if "dix" in text.lower() and "date" in text.lower():
                    # Try to extract JSON data
                    import re
                    json_match = re.search(r'\[[\s\S]*?\{[\s\S]*?"date"[\s\S]*?\}[\s\S]*?\]', text)
                    if json_match:
                        data = json.loads(json_match.group())
                        df = pd.DataFrame(data)
                        if "date" in df.columns:
                            df["date"] = pd.to_datetime(df["date"])
                            df = df.set_index("date").sort_index()
                        print(f"    Scraped {len(df)} rows from page")
                        if save:
                            path = DATA_RAW / "squeezemetrics_dix_gex.csv"
                            df.to_csv(path)
                        return df
            print("    Could not find data in page HTML")
    except Exception as e:
        print(f"    Page scraping failed: {e}")

    print("    WARNING: Could not collect SqueezeMetrics data automatically.")
    print("    Manual download: https://squeezemetrics.com/monitor/dix → Download CSV")
    return pd.DataFrame()


def collect_stockgrid_darkpool(ticker: str = TARGET_TICKER, save: bool = True) -> pd.DataFrame:
    """
    Scrape Stockgrid dark pool data for a specific ticker.
    Stockgrid shows net short dollar volume from FINRA dark pools.
    """
    print(f"  Collecting Stockgrid dark pool data for {ticker}...")

    url = f"https://stockgrid.io/darkpools/{ticker}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")

            # Look for data in script tags or API calls
            scripts = soup.find_all("script")
            for script in scripts:
                text = script.string or ""
                if ticker.lower() in text.lower() and ("volume" in text.lower() or "short" in text.lower()):
                    import re
                    json_match = re.search(r'\[[\s\S]*?\{[\s\S]*?"date"[\s\S]*?\}[\s\S]*?\]', text)
                    if json_match:
                        data = json.loads(json_match.group())
                        df = pd.DataFrame(data)
                        print(f"    Got {len(df)} rows from Stockgrid")
                        if save:
                            path = DATA_RAW / f"stockgrid_{ticker}_darkpool.csv"
                            df.to_csv(path, index=False)
                        return df
    except Exception as e:
        print(f"    Stockgrid scraping failed: {e}")

    # Try Stockgrid API endpoint
    try:
        api_url = f"https://stockgrid.io/get_dark_pool_individual_data"
        params = {"ticker": ticker}
        resp = requests.get(api_url, params=params, headers=headers, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                df = pd.DataFrame(data)
                print(f"    Got {len(df)} rows from Stockgrid API")
                if save:
                    path = DATA_RAW / f"stockgrid_{ticker}_darkpool.csv"
                    df.to_csv(path, index=False)
                return df
            elif isinstance(data, dict):
                # Try different data structure
                for key in data:
                    if isinstance(data[key], list):
                        df = pd.DataFrame(data[key])
                        print(f"    Got {len(df)} rows from Stockgrid API (key={key})")
                        if save:
                            path = DATA_RAW / f"stockgrid_{ticker}_darkpool.csv"
                            df.to_csv(path, index=False)
                        return df
    except Exception as e:
        print(f"    Stockgrid API failed: {e}")

    print(f"    WARNING: Could not scrape Stockgrid data for {ticker}.")
    print(f"    Manual: Visit https://stockgrid.io/darkpools/{ticker}")
    return pd.DataFrame()


def collect_finra_short_volume(ticker: str = TARGET_TICKER, save: bool = True) -> pd.DataFrame:
    """
    Alternative: Get short volume data from FINRA (public daily data).
    Uses the Quandl/FINRA short interest endpoint or chart exchange.
    """
    print(f"  Collecting FINRA short volume for {ticker}...")

    # Try chartexchange.com (provides historical short volume)
    url = f"https://chartexchange.com/symbol/nyse-{ticker.lower()}/short-volume/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code == 200:
            tables = pd.read_html(resp.text)
            if tables:
                df = tables[0]
                print(f"    Got {len(df)} rows from ChartExchange")
                if save:
                    path = DATA_RAW / f"finra_short_volume_{ticker}.csv"
                    df.to_csv(path, index=False)
                return df
    except Exception as e:
        print(f"    ChartExchange failed: {e}")

    print(f"    Could not get FINRA short volume for {ticker}.")
    return pd.DataFrame()


if __name__ == "__main__":
    print("=" * 60)
    print("DARK POOL DATA COLLECTION")
    print("=" * 60)
    dix = collect_squeezemetrics_dix()
    sg = collect_stockgrid_darkpool()
    finra = collect_finra_short_volume()
    print("\nDone.")
