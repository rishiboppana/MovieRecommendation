"""
Load precomputed ALS recommendations from parquet and push to Redis.

Run this after training to warm the cache before starting the API.

Usage:
    python serving/redis_precompute.py
    python serving/redis_precompute.py --ttl 86400 --batch-size 5000
"""
import os
import json
import argparse
import redis
import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
DEFAULT_TTL = 86400  # 24 hours


def load_recommendations(path: Path) -> pd.DataFrame:
    """Load precomputed recs parquet → flat DataFrame."""
    if not path.exists():
        raise FileNotFoundError(f"Run 'make train' first: {path}")
    table = pq.read_table(str(path))
    df = table.to_pandas()
    print(f"Loaded {len(df):,} user recommendation sets")
    return df




def push_recommendations(df: pd.DataFrame, r: redis.Redis, ttl: int, batch_size: int):
    """Write user recs to Redis: key=recs:{user_id}, value=JSON list."""
    pipe = r.pipeline(transaction=False)
    count = 0

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Pushing user recs"):
        user_id = int(row["user_id"])
        recs = row["recommendations"]  # list of Row(movie_id, rating)
        if recs is None:
            continue
        payload = [{"movie_id": int(rec["movie_id"]), "score": float(rec["rating"])} for rec in recs]
        pipe.setex(f"recs:{user_id}", ttl, json.dumps(payload))
        count += 1

        if count % batch_size == 0:
            pipe.execute()
            pipe = r.pipeline(transaction=False)

    pipe.execute()
    print(f"  Pushed {count:,} user recommendation sets")


def push_similar_from_factors(factors_path: Path, r: redis.Redis, ttl: int, top_k: int = 10):
    """
    Compute item-to-item cosine similarity from ALS item factor vectors,
    then push top-K similar movies per movie to Redis.
    Uses sklearn NearestNeighbors — no Spark needed.
    """
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    if not factors_path.exists():
        print(f"  Item factors not found at {factors_path}, skipping similarity.")
        return

    print("Computing item-to-item similarity from ALS factors...")
    factor_files = list(factors_path.glob("*.parquet"))
    if not factor_files:
        print("  No factor parquet files found.")
        return

    frames = [pd.read_parquet(str(f)) for f in factor_files]
    factors_df = pd.concat(frames, ignore_index=True).sort_values("id").reset_index(drop=True)
    movie_ids = factors_df["id"].values
    # features column is a list/array of floats
    factor_matrix = np.stack(factors_df["features"].values)

    print(f"  Factor matrix: {factor_matrix.shape} ({len(movie_ids):,} movies × {factor_matrix.shape[1]} dims)")

    # Cosine similarity via NearestNeighbors (metric='cosine', brute force)
    # For 59K items this takes ~2 min — acceptable for a one-time precompute
    nn = NearestNeighbors(n_neighbors=top_k + 1, metric="cosine", algorithm="brute", n_jobs=-1)
    nn.fit(factor_matrix)
    distances, indices = nn.kneighbors(factor_matrix)

    pipe = r.pipeline(transaction=False)
    count = 0
    for i, movie_id in enumerate(tqdm(movie_ids, desc="Pushing similar movies")):
        # Skip first neighbor (itself, distance ≈ 0)
        neighbor_idx = indices[i][1:]
        neighbor_dist = distances[i][1:]
        payload = [
            {"movie_id": int(movie_ids[j]), "score": round(1 - float(d), 4)}
            for j, d in zip(neighbor_idx, neighbor_dist)
        ]
        pipe.setex(f"similar:{int(movie_id)}", ttl, json.dumps(payload))
        count += 1
        if count % 5000 == 0:
            pipe.execute()
            pipe = r.pipeline(transaction=False)

    pipe.execute()
    print(f"  Pushed {count:,} item similarity sets")


def push_popular(movies_path: Path, r: redis.Redis, ttl: int):
    """Cache top-1000 popular movies for cold-start."""
    if not movies_path.exists():
        print("  Skipping popular movies (no movie_features)")
        return
    df = pq.read_table(str(movies_path)).to_pandas()
    top = (
        df.sort_values("rating_count", ascending=False)
        .head(1000)
    )
    payload = [
        {
            "movie_id": int(r_["movie_id"]),
            "title": str(r_.get("title", "")),
            "genres": str(r_.get("genres", "")),
            "score": float(r_.get("avg_rating", 0)) if pd.notna(r_.get("avg_rating")) else 0.0,
            "poster_url": r_.get("poster_url") if pd.notna(r_.get("poster_url")) else None,
            "avg_rating": float(r_.get("avg_rating", 0)) if pd.notna(r_.get("avg_rating")) else None,
            "rating_count": int(r_["rating_count"]) if pd.notna(r_.get("rating_count")) else 0,
        }
        for _, r_ in top.iterrows()
    ]
    r.setex("popular_movies", ttl, json.dumps(payload))
    print(f"  Cached {len(payload)} popular movies")


def main(ttl: int = DEFAULT_TTL, batch_size: int = 5000):
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    try:
        r.ping()
        print(f"Redis connected: {REDIS_HOST}:{REDIS_PORT}")
    except redis.ConnectionError:
        print(f"ERROR: Cannot connect to Redis at {REDIS_HOST}:{REDIS_PORT}")
        return

    recs_path = ROOT / "data" / "recommendations"
    item_factors_path = ROOT / "mlflow" / "artifacts" / "als-model-local" / "itemFactors"
    movies_path = ROOT / "data" / "processed" / "movie_features"

    print("\n=== Pushing precomputed recommendations to Redis ===")
    recs_df = load_recommendations(recs_path)
    push_recommendations(recs_df, r, ttl, batch_size)

    push_similar_from_factors(item_factors_path, r, ttl)

    push_popular(movies_path, r, ttl)

    # Print cache stats
    total_keys = r.dbsize()
    print(f"\nRedis cache: {total_keys:,} total keys | TTL: {ttl}s")
    r.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ttl", type=int, default=DEFAULT_TTL, help="Cache TTL in seconds")
    parser.add_argument("--batch-size", type=int, default=5000, help="Pipeline batch size")
    args = parser.parse_args()
    main(ttl=args.ttl, batch_size=args.batch_size)
