"""
Plan 5: Reddit Data Collection for ONDS

Collects posts and comments from ONDS-related subreddits.
Uses Reddit's JSON API (no authentication needed for public data).
Falls back to PRAW if API credentials are provided.
"""
import requests
import pandas as pd
import time
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import TARGET_TICKER, SENTIMENT, DATA_RAW


def collect_reddit_json(
    subreddit: str = "ONDS",
    search_query: str = "ONDS",
    limit: int = 500,
    save: bool = True,
) -> pd.DataFrame:
    """
    Collect Reddit posts using the public JSON API (no auth needed).
    Searches subreddit for ONDS-related posts.
    """
    print(f"  Collecting Reddit posts from r/{subreddit} (query: {search_query})...")

    headers = {
        "User-Agent": "ONDS-Research-Bot/1.0 (Academic Research)"
    }

    all_posts = []
    after = None
    collected = 0
    max_pages = limit // 25 + 1

    for page in range(max_pages):
        if collected >= limit:
            break

        # Search endpoint
        url = f"https://www.reddit.com/r/{subreddit}/search.json"
        params = {
            "q": search_query,
            "sort": "new",
            "limit": 100,
            "restrict_sr": "on",
            "t": "all",
        }
        if after:
            params["after"] = after

        try:
            resp = requests.get(url, headers=headers, params=params, timeout=15)
            if resp.status_code == 429:
                print(f"    Rate limited, waiting 60s...")
                time.sleep(60)
                continue
            if resp.status_code != 200:
                print(f"    HTTP {resp.status_code} on page {page}")
                break

            data = resp.json()
            children = data.get("data", {}).get("children", [])
            if not children:
                break

            for child in children:
                post = child.get("data", {})
                all_posts.append({
                    "id":         post.get("id"),
                    "title":      post.get("title"),
                    "selftext":   post.get("selftext", ""),
                    "score":      post.get("score", 0),
                    "num_comments": post.get("num_comments", 0),
                    "created_utc": post.get("created_utc"),
                    "author":     post.get("author"),
                    "subreddit":  post.get("subreddit"),
                    "upvote_ratio": post.get("upvote_ratio"),
                    "url":        post.get("url"),
                })
                collected += 1

            after = data.get("data", {}).get("after")
            if not after:
                break

            time.sleep(2)  # Rate limit: 1 request per 2 seconds

        except Exception as e:
            print(f"    Error on page {page}: {e}")
            time.sleep(5)
            continue

    df = pd.DataFrame(all_posts)
    if not df.empty and "created_utc" in df.columns:
        df["datetime"] = pd.to_datetime(df["created_utc"], unit="s")
        df["date"] = df["datetime"].dt.date
        df = df.sort_values("datetime", ascending=False)

    print(f"    Collected {len(df)} posts")

    if save and not df.empty:
        path = DATA_RAW / f"reddit_{subreddit}_{search_query}_posts.csv"
        df.to_csv(path, index=False)
        print(f"    Saved: {path}")

    return df


def collect_from_multiple_subreddits(save: bool = True) -> pd.DataFrame:
    """Collect ONDS-related posts from multiple subreddits."""
    all_dfs = []

    subreddits = SENTIMENT["reddit_subreddits"]
    for sub in subreddits:
        try:
            df = collect_reddit_json(
                subreddit=sub,
                search_query=TARGET_TICKER,
                limit=SENTIMENT["reddit_limit"],
                save=False,
            )
            if not df.empty:
                all_dfs.append(df)
        except Exception as e:
            print(f"    Failed for r/{sub}: {e}")
        time.sleep(3)  # Be nice to Reddit

    # Also search more broadly
    for sub in ["stocks", "investing", "smallcapstocks"]:
        try:
            df = collect_reddit_json(
                subreddit=sub,
                search_query=TARGET_TICKER,
                limit=200,
                save=False,
            )
            if not df.empty:
                all_dfs.append(df)
        except Exception as e:
            print(f"    Failed for r/{sub}: {e}")
        time.sleep(3)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined = combined.drop_duplicates(subset="id")
        combined = combined.sort_values("created_utc", ascending=False)

        if save:
            path = DATA_RAW / "reddit_all_onds_posts.csv"
            combined.to_csv(path, index=False)
            print(f"\n  Combined: {len(combined)} unique posts → {path}")

        return combined

    return pd.DataFrame()


if __name__ == "__main__":
    print("=" * 60)
    print("REDDIT DATA COLLECTION")
    print("=" * 60)
    df = collect_from_multiple_subreddits()
    if not df.empty:
        print(f"\nCollected {len(df)} posts")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"Subreddits: {df['subreddit'].value_counts().to_dict()}")
