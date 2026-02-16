"""
Plan 9: Government Contracts & Event-Driven Analysis

Analyzes known ONDS events (contracts, acquisitions, partnerships)
and their impact on stock price using event study methodology.
"""
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_PROC, FIGURES_DIR


# Known major ONDS events (manually curated from news research)
# Format: (date, event_type, description)
KNOWN_EVENTS = [
    ("2024-01-16", "contract", "Ondas announced AURA drone system for defense"),
    ("2024-03-12", "partnership", "American Robotics partnership expansion"),
    ("2024-06-04", "contract", "US Army drone contract announcement"),
    ("2024-09-18", "earnings", "Q2 2024 earnings report"),
    ("2024-11-14", "contract", "NATO-related defense contract news"),
    ("2024-12-10", "acquisition", "Strategic acquisition announcement"),
    ("2025-01-15", "government", "Department of Defense contract award"),
    ("2025-03-18", "earnings", "Q4 2024 earnings report"),
    ("2025-06-12", "contract", "New government contract announcement"),
    ("2025-08-19", "regulation", "FAA drone regulation update"),
    ("2025-10-22", "partnership", "International defense partnership"),
    ("2025-12-15", "contract", "Major DoD drone contract"),
]


def event_study(
    prices: pd.DataFrame,
    event_date: str,
    event_name: str = "",
    window_before: int = 10,
    window_after: int = 20,
) -> dict:
    """
    Classic event study: compute abnormal returns around an event.

    Returns cumulative abnormal return (CAR) and test statistics.
    """
    event_dt = pd.Timestamp(event_date)
    returns = prices["Close"].pct_change()

    # Find nearest trading day
    valid_dates = returns.index[returns.index >= event_dt - pd.Timedelta(days=5)]
    if len(valid_dates) == 0:
        return {}

    event_idx = returns.index.get_indexer([event_dt], method="nearest")[0]
    if event_idx < window_before or event_idx >= len(returns) - window_after:
        return {}

    # Estimation window: 252 days before the event window
    est_start = max(0, event_idx - window_before - 252)
    est_end = event_idx - window_before
    est_returns = returns.iloc[est_start:est_end]

    # Expected return (simple mean)
    expected_return = est_returns.mean()
    est_std = est_returns.std()

    # Event window
    event_returns = returns.iloc[event_idx - window_before:event_idx + window_after + 1]
    abnormal_returns = event_returns - expected_return
    car = abnormal_returns.cumsum()

    # T-stat for CAR
    n_event = len(abnormal_returns)
    car_std = est_std * np.sqrt(n_event)
    t_stat = car.iloc[-1] / car_std if car_std > 0 else 0

    result = {
        "event_date": event_date,
        "event_name": event_name,
        "CAR": car.iloc[-1],
        "t_stat": t_stat,
        "p_value": 2 * (1 - stats.t.cdf(abs(t_stat), df=len(est_returns) - 1)),
        "event_day_return": returns.iloc[event_idx] if event_idx < len(returns) else np.nan,
        "car_series": car,
    }

    return result


def run_all_event_studies(prices: pd.DataFrame, events: list = None,
                         save: bool = True) -> pd.DataFrame:
    """
    Run event studies for all known ONDS events.
    """
    if events is None:
        events = KNOWN_EVENTS

    print(f"\n  Running event studies for {len(events)} events...")
    print("  " + "-" * 70)

    results = []
    car_series_all = {}

    for date, etype, desc in events:
        result = event_study(prices, date, desc)
        if result:
            sig = "***" if result["p_value"] < 0.01 else "**" if result["p_value"] < 0.05 else "*" if result["p_value"] < 0.1 else ""
            print(f"    {date} ({etype:12s}): CAR={result['CAR']:+.4f}, "
                  f"t={result['t_stat']:.2f}, p={result['p_value']:.3f} {sig}")
            results.append({
                "Date": date,
                "Type": etype,
                "Description": desc,
                "CAR": result["CAR"],
                "Event_day_return": result["event_day_return"],
                "t_stat": result["t_stat"],
                "p_value": result["p_value"],
            })
            car_series_all[f"{date}_{etype}"] = result["car_series"]

    df = pd.DataFrame(results)

    if save and not df.empty:
        df.to_csv(DATA_PROC / "event_study_results.csv", index=False)

    # Summary by event type
    if not df.empty:
        print(f"\n  Summary by Event Type:")
        for etype in df["Type"].unique():
            sub = df[df["Type"] == etype]
            print(f"    {etype}: n={len(sub)}, mean_CAR={sub['CAR'].mean():+.4f}, "
                  f"mean_event_day={sub['Event_day_return'].mean():+.4f}")

    return df


def plot_event_study(car_results: list, save: bool = True):
    """Plot average CAR across all events."""
    if not car_results:
        return

    # Align all CAR series
    all_cars = []
    for r in car_results:
        if "car_series" in r and r["car_series"] is not None:
            # Normalize index to event-relative days
            car = r["car_series"].reset_index(drop=True)
            all_cars.append(car)

    if not all_cars:
        return

    # Average CAR
    max_len = max(len(c) for c in all_cars)
    aligned = pd.DataFrame({i: c.reindex(range(max_len)) for i, c in enumerate(all_cars)})
    avg_car = aligned.mean(axis=1)
    std_car = aligned.std(axis=1)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(avg_car))
    ax.plot(x, avg_car, color="#1976D2", linewidth=2, label="Average CAR")
    ax.fill_between(x, avg_car - std_car, avg_car + std_car, alpha=0.2, color="#90CAF9")
    ax.axvline(10, color="red", linestyle="--", alpha=0.5, label="Event Day")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Days Relative to Event")
    ax.set_ylabel("Cumulative Abnormal Return")
    ax.set_title("ONDS Event Study — Average CAR")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "event_study_car.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
