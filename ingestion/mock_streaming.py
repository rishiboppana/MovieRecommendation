"""
Mock streaming simulator — no Kafka or Docker required.

Simulates Spark Structured Streaming output by writing realistic trending
data directly to Redis every 30 seconds. Mimics what the Spark Streaming
consumer would produce from a 5-minute tumbling window over user events.

Usage:
    python ingestion/mock_streaming.py          # runs indefinitely
    python ingestion/mock_streaming.py --once   # single update then exit
"""
import os
import json
import time
import random
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import redis
import pyarrow.parquet as pq
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
TRENDING_KEY = "trending:realtime"
TRENDING_TTL = 300      # 5-minute window, matching Spark config
UPDATE_INTERVAL = 30    # seconds between window refreshes


def load_candidate_movies(n: int = 500) -> list[dict]:
    """Load top-N popular movies as the event candidate pool."""
    path = ROOT / "data" / "processed" / "movie_features"
    if not path.exists():
        raise FileNotFoundError("Run 'make etl' first — movie_features not found")
    df = pq.read_table(str(path)).to_pandas()
    top = (
        df[df["rating_count"] >= 100]
        .sort_values("rating_count", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )
    return top.to_dict("records")


def simulate_window(movies: list[dict], n_top: int = 20, window_minutes: int = 5) -> list[dict]:
    """
    Simulate events in a tumbling window and return top-N trending movies.

    Uses a realistic distribution:
    - Base engagement proportional to sqrt(rating_count) (popularity prior)
    - Multiplied by a random "trending boost" (1x to 5x) per movie per window
    - Poisson-sampled event counts to mimic real traffic
    """
    rng = np.random.default_rng()

    # Base weights: popularity prior (sqrt dampens dominance of mega-popular films)
    counts = np.array([float(m.get("rating_count") or 1) for m in movies])
    base_weights = np.sqrt(counts)
    base_weights /= base_weights.sum()

    # Trending boosts: random per window so ranking shifts each refresh
    boosts = rng.uniform(0.5, 5.0, size=len(movies))
    final_weights = base_weights * boosts
    final_weights /= final_weights.sum()

    # Expected events in a 5-minute window at ~50 events/sec = 15,000 total events
    total_events = int(rng.poisson(window_minutes * 60 * 50))
    # Multinomial draw: distribute events across movies
    event_counts = rng.multinomial(total_events, final_weights)

    # Build result: top-N by event count
    indexed = [(int(event_counts[i]), movies[i]) for i in range(len(movies))]
    indexed.sort(key=lambda x: x[0], reverse=True)

    return [
        {"movie_id": int(m["movie_id"]), "event_count": cnt}
        for cnt, m in indexed[:n_top]
        if cnt > 0
    ]


def push_trending(r: redis.Redis, trending: list[dict]) -> None:
    r.setex(TRENDING_KEY, TRENDING_TTL, json.dumps(trending))


def run(once: bool = False) -> None:
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    try:
        r.ping()
        print(f"Redis connected: {REDIS_HOST}:{REDIS_PORT}")
    except redis.ConnectionError as e:
        raise SystemExit(f"Cannot connect to Redis: {e}")

    print("Loading candidate movies…")
    movies = load_candidate_movies(n=500)
    print(f"Loaded {len(movies)} candidate movies")

    print(f"Mock streaming started — updating every {UPDATE_INTERVAL}s  (Ctrl+C to stop)\n")

    tick = 0
    try:
        while True:
            tick += 1
            trending = simulate_window(movies, n_top=20)
            push_trending(r, trending)

            top3 = trending[:3]
            ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
            print(
                f"[{ts}] Window #{tick:03d} | "
                + "  ".join(
                    f"{m['movie_id']}×{m['event_count']}"
                    for m in top3
                )
                + f"  (+{len(trending)-3} more)"
            )

            if once:
                break
            time.sleep(UPDATE_INTERVAL)

    except KeyboardInterrupt:
        print("\nMock streaming stopped.")
    finally:
        r.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run one update then exit")
    args = parser.parse_args()
    run(once=args.once)
