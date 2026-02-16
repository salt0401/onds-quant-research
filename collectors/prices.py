"""
Plan 1/2/3: Collect price data for ONDS, peers, sector ETFs, and cross-assets via yfinance.
This is the foundation — every other analysis depends on this data.
"""
import yfinance as yf
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ALL_TICKERS, TARGET_TICKER, DATA_RAW, DATA_START, DATA_END


def download_all_prices(
    tickers: list = None,
    start: str = DATA_START,
    end: str = DATA_END,
    save: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Download OHLCV data for all tickers.
    Returns dict of {ticker: DataFrame} with columns [Open, High, Low, Close, Volume].
    """
    if tickers is None:
        tickers = ALL_TICKERS

    data = {}
    failed = []

    for ticker in tickers:
        print(f"  Downloading {ticker}...", end=" ")
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if df.empty:
                print("EMPTY")
                failed.append(ticker)
                continue

            # Flatten multi-level columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.index = pd.to_datetime(df.index)
            df.index.name = "Date"
            data[ticker] = df
            print(f"OK ({len(df)} rows, {df.index[0].date()} to {df.index[-1].date()})")

            if save:
                path = DATA_RAW / f"{ticker.replace('^', '').replace('-', '_')}_prices.csv"
                df.to_csv(path)
        except Exception as e:
            print(f"FAILED: {e}")
            failed.append(ticker)

    if failed:
        print(f"\n  WARNING: Failed tickers: {failed}")

    # Also create a combined close-price matrix
    closes = pd.DataFrame({t: d["Close"] for t, d in data.items()})
    if save:
        closes.to_csv(DATA_RAW / "all_closes.csv")
        print(f"  Saved combined close prices: {DATA_RAW / 'all_closes.csv'}")

    return data


def load_prices(ticker: str = TARGET_TICKER) -> pd.DataFrame:
    """Load previously saved price data."""
    path = DATA_RAW / f"{ticker.replace('^', '').replace('-', '_')}_prices.csv"
    if not path.exists():
        raise FileNotFoundError(f"No saved prices for {ticker}. Run download_all_prices() first.")
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    return df


def load_all_closes() -> pd.DataFrame:
    """Load the combined close-price matrix."""
    path = DATA_RAW / "all_closes.csv"
    if not path.exists():
        raise FileNotFoundError("No combined close prices. Run download_all_prices() first.")
    return pd.read_csv(path, index_col="Date", parse_dates=True)


if __name__ == "__main__":
    print("=" * 60)
    print("PRICE DATA COLLECTION")
    print("=" * 60)
    data = download_all_prices()
    print(f"\nSuccessfully downloaded {len(data)} tickers.")
    print(f"ONDS: {len(data.get('ONDS', []))} trading days")
