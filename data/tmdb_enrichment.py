"""
Enrich MovieLens movies with TMDB metadata: poster URLs, genres, overview, popularity.
Reads MovieLens links.csv (movieId → tmdbId) and writes enriched parquet.

Usage:
    python data/tmdb_enrichment.py
    TMDB_API_KEY=xxx python data/tmdb_enrichment.py --limit 1000
"""
import os
import time
import json
import argparse
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).parent
TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w342"
CACHE_FILE = DATA_DIR / "tmdb_cache.json"


def get_tmdb_movie(tmdb_id: int, api_key: str, cache: dict) -> dict | None:
    key = str(tmdb_id)
    if key in cache:
        return cache[key]

    url = f"{TMDB_BASE}/movie/{tmdb_id}"
    try:
        resp = requests.get(url, params={"api_key": api_key}, timeout=10)
        if resp.status_code == 404:
            cache[key] = None
            return None
        if resp.status_code == 429:
            time.sleep(2)
            return get_tmdb_movie(tmdb_id, api_key, cache)
        resp.raise_for_status()
        data = resp.json()
        result = {
            "tmdb_id": tmdb_id,
            "title": data.get("title"),
            "overview": data.get("overview"),
            "release_date": data.get("release_date"),
            "popularity": data.get("popularity"),
            "vote_average": data.get("vote_average"),
            "vote_count": data.get("vote_count"),
            "genres": [g["name"] for g in data.get("genres", [])],
            "poster_url": f"{TMDB_IMG_BASE}{data['poster_path']}" if data.get("poster_path") else None,
            "runtime": data.get("runtime"),
            "original_language": data.get("original_language"),
        }
        cache[key] = result
        return result
    except Exception as e:
        print(f"  Error fetching tmdb_id={tmdb_id}: {e}")
        cache[key] = None
        return None


def main(limit: int | None = None):
    api_key = os.getenv("TMDB_API_KEY")
    if not api_key:
        print("ERROR: Set TMDB_API_KEY in .env")
        return

    links_path = DATA_DIR / "movielens" / "ml-25m" / "links.csv"
    if not links_path.exists():
        print(f"ERROR: {links_path} not found. Run download_datasets.py first.")
        return

    links = pd.read_csv(links_path)
    links = links.dropna(subset=["tmdbId"])
    links["tmdbId"] = links["tmdbId"].astype(int)

    if limit:
        links = links.head(limit)

    # Load existing cache
    cache = {}
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            cache = json.load(f)
    print(f"Cache: {len(cache)} entries loaded")

    records = []
    for _, row in tqdm(links.iterrows(), total=len(links), desc="TMDB enrichment"):
        result = get_tmdb_movie(int(row["tmdbId"]), api_key, cache)
        if result:
            result["movie_id"] = int(row["movieId"])
            records.append(result)
        time.sleep(0.05)  # ~20 req/s, well within free tier limit

    # Save cache
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)

    if not records:
        print("No records fetched.")
        return

    df = pd.DataFrame(records)
    df["genres_str"] = df["genres"].apply(lambda g: "|".join(g) if g else "")
    out_path = DATA_DIR / "enriched" / "tmdb_metadata.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df)} movies to {out_path}")
    print(df[["movie_id", "title", "genres_str", "popularity"]].head(10))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Max movies to fetch (default: all)")
    args = parser.parse_args()
    main(limit=args.limit)
