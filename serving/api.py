"""
FastAPI serving layer for the Movie Recommendation System.

Endpoints:
  GET  /recommend/{user_id}     → top-N personalized picks (Redis cache)
  GET  /similar/{movie_id}      → item-to-item similar movies
  POST /rate                    → ingest new rating → Kafka
  GET  /movies/search           → full-text movie search
  GET  /trending                → trending movies (real-time from Kafka stream)
  GET  /movies/{movie_id}       → movie detail
  GET  /health                  → health check
"""
import os
import json
import asyncio
from typing import Optional
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import redis.asyncio as aioredis
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import pandas as pd
import pyarrow.parquet as pq
from dotenv import load_dotenv

load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
KAFKA_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))

# ─────────────────────────── Models ───────────────────────────

class RatingRequest(BaseModel):
    user_id: int = Field(..., ge=1)
    movie_id: int = Field(..., ge=1)
    rating: float = Field(..., ge=0.5, le=5.0)

    @field_validator("rating")
    @classmethod
    def round_rating(cls, v):
        return round(v * 2) / 2  # snap to nearest 0.5


class RecommendedMovie(BaseModel):
    movie_id: int
    title: str
    score: float
    genres: str
    poster_url: Optional[str] = None
    avg_rating: Optional[float] = None
    rating_count: Optional[int] = None


class LikedMovie(BaseModel):
    movie_id: int
    weight: float = Field(1.0, ge=0.1, le=5.0)


class LikesRequest(BaseModel):
    likes: list[LikedMovie] = Field(..., min_length=1, max_length=50)
    n: int = Field(10, ge=1, le=50)


class SimilarMovie(BaseModel):
    movie_id: int
    title: str
    score: float
    genres: str
    poster_url: Optional[str] = None


# ─────────────────────────── Startup ──────────────────────────

_redis: aioredis.Redis | None = None
_kafka: KafkaProducer | None = None
_movies_df: pd.DataFrame | None = None
_movies_with_meta: set[int] = set()      # movie_ids that have real metadata

# Item-factor matrix for centroid-based taste recommendations
_item_factors: np.ndarray | None = None  # (n_movies, rank), L2-normalised
_item_ids: np.ndarray | None = None      # int32, parallel to _item_factors rows
_id_to_idx: dict[int, int] = {}          # movie_id → row index


def load_movies_df():
    """Load movie metadata once at startup for search/detail endpoints."""
    global _movies_df, _movies_with_meta
    path = DATA_DIR / "processed" / "movie_features"
    if path.exists():
        _movies_df = pq.read_table(str(path)).to_pandas()
        _movies_with_meta = set(_movies_df["movie_id"].astype(int).tolist())
        print(f"Loaded {len(_movies_df):,} movies from feature store")
    else:
        print("WARNING: movie_features not found — run ETL first")
        _movies_df = pd.DataFrame(columns=["movie_id", "title", "genres", "avg_rating", "rating_count", "poster_url"])


def load_item_factors():
    """
    Load ALS item-factor vectors into memory and L2-normalise them so that
    dot-product == cosine similarity.  Tries the Docker mount path first,
    then the local project path.
    """
    global _item_factors, _item_ids, _id_to_idx
    candidates = [
        Path("/mlflow/artifacts/als-model-local/itemFactors"),
        Path("./mlflow/artifacts/als-model-local/itemFactors"),
    ]
    factors_path = next((p for p in candidates if p.exists()), None)
    if factors_path is None:
        print("WARNING: Item factors not found — from-likes will use similarity fallback")
        return

    frames = [pq.read_table(str(f)).to_pandas() for f in sorted(factors_path.glob("*.parquet"))]
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True).sort_values("id").reset_index(drop=True)

    _item_ids = df["id"].values.astype(np.int32)
    mat = np.stack(df["features"].values).astype(np.float32)

    # L2-normalise each row so cosine sim = dot product
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    _item_factors = mat / norms

    _id_to_idx = {int(mid): i for i, mid in enumerate(_item_ids)}
    print(f"Item factors loaded: {_item_factors.shape}  ({_item_factors.shape[1]}-dim ALS embeddings)")


def init_kafka():
    global _kafka
    try:
        _kafka = KafkaProducer(
            bootstrap_servers=KAFKA_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: str(k).encode("utf-8"),
        )
        print(f"Kafka connected: {KAFKA_SERVERS}")
    except NoBrokersAvailable:
        print(f"WARNING: Kafka not available ({KAFKA_SERVERS}) — ratings won't stream")
        _kafka = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _redis
    _redis = aioredis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    await _redis.ping()
    print(f"Redis connected: {REDIS_HOST}:{REDIS_PORT}")
    load_movies_df()
    load_item_factors()
    init_kafka()
    yield
    await _redis.aclose()
    if _kafka:
        _kafka.close()


# ─────────────────────────── App ──────────────────────────────

app = FastAPI(
    title="Movie Recommendation API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────── Helpers ──────────────────────────

def _solve_user_vector(likes: list[LikedMovie], reg: float = 0.1) -> np.ndarray | None:
    """
    Solve for the optimal user latent-factor vector using the ALS update rule.

    Given liked movies with confidence weights, we solve:

        u* = argmin_u  Σ_i c_i (p_i - u · x_i)^2  +  λ ||u||^2

    where:
        x_i = item factor vector for liked movie i   (from pre-trained ALS)
        p_i = 1  (user prefers this movie)
        c_i = 1 + α * weight_i  (confidence; higher weight → stronger signal)
        λ   = regularisation (prevents overfitting when only a few likes exist)

    Closed-form solution:
        u* = (X^T C X + λ I)^{-1}  X^T c

    This is the exact same formula ALS uses to update user vectors during
    training — so each "like" is a genuine micro-training step on the user.
    More likes → better-conditioned system → sharper taste profile.
    """
    if _item_factors is None or _item_ids is None:
        return None

    alpha = 25.0                                # matches training alpha
    rows, weights = [], []
    for lk in likes:
        idx = _id_to_idx.get(lk.movie_id)
        if idx is None:
            continue
        rows.append(_item_factors[idx])         # already L2-normalised row
        weights.append(1.0 + alpha * lk.weight)

    if not rows:
        return None

    X = np.stack(rows)                          # (n_likes, rank)
    c = np.array(weights)                       # (n_likes,)
    rank = X.shape[1]

    XtC = X.T * c                               # (rank, n_likes)
    XtCX = XtC @ X + reg * np.eye(rank)        # (rank, rank)
    XtCp = XtC @ np.ones(len(rows))            # p_i = 1 for all likes

    try:
        u = np.linalg.solve(XtCX, XtCp)        # (rank,)
    except np.linalg.LinAlgError:
        u = X.mean(axis=0)                      # fallback to centroid

    norm = np.linalg.norm(u)
    if norm > 0:
        u /= norm
    return u


def _recommend_from_user_vector(u: np.ndarray, liked_set: set[int], n: int) -> list[dict]:
    """Score all items by dot product with the user vector, filter, return top-N."""
    scores = _item_factors @ u                  # cosine sim (both normalised)

    budget = min(n * 6, len(scores))
    top_idx = np.argpartition(scores, -budget)[-budget:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

    results = []
    for idx in top_idx:
        mid = int(_item_ids[idx])
        if mid in liked_set or mid not in _movies_with_meta:
            continue
        results.append({**get_movie_meta(mid), "score": round(float(scores[idx]), 4)})
        if len(results) >= n:
            break
    return results


def get_movie_meta(movie_id: int) -> dict:
    if _movies_df is None or _movies_df.empty:
        return {"movie_id": movie_id, "title": f"Movie {movie_id}", "genres": "", "poster_url": None}
    row = _movies_df[_movies_df["movie_id"] == movie_id]
    if row.empty:
        return {"movie_id": movie_id, "title": f"Movie {movie_id}", "genres": "", "poster_url": None}
    r = row.iloc[0]
    return {
        "movie_id": int(r["movie_id"]),
        "title": str(r.get("title", "")),
        "genres": str(r.get("genres", "")),
        "poster_url": r.get("poster_url") if pd.notna(r.get("poster_url")) else None,
        "avg_rating": float(r["avg_rating"]) if pd.notna(r.get("avg_rating")) else None,
        "rating_count": int(r["rating_count"]) if pd.notna(r.get("rating_count")) else None,
    }


async def get_popular_movies(n: int = 10) -> list[dict]:
    """Fallback for cold-start: return most popular movies."""
    cached = await _redis.get("popular_movies")
    if cached:
        return json.loads(cached)[:n]
    if _movies_df is not None and not _movies_df.empty:
        top = (
            _movies_df
            .sort_values("rating_count", ascending=False)
            .head(n)
        )
        result = []
        for _, row in top.iterrows():
            result.append({
                "movie_id": int(row["movie_id"]),
                "title": str(row.get("title", "")),
                "score": float(row.get("avg_rating", 0)),
                "genres": str(row.get("genres", "")),
                "poster_url": row.get("poster_url") if pd.notna(row.get("poster_url")) else None,
                "avg_rating": float(row.get("avg_rating", 0)) if pd.notna(row.get("avg_rating")) else None,
                "rating_count": int(row["rating_count"]) if pd.notna(row.get("rating_count")) else None,
            })
        return result
    return []


# ─────────────────────────── Endpoints ────────────────────────

@app.get("/health")
async def health():
    redis_ok = False
    try:
        redis_ok = await _redis.ping()
    except Exception:
        pass
    return {
        "status": "ok",
        "redis": redis_ok,
        "kafka": _kafka is not None,
        "movies_loaded": len(_movies_df) if _movies_df is not None else 0,
    }


@app.get("/recommend/{user_id}", response_model=list[RecommendedMovie])
async def recommend(user_id: int, n: int = Query(10, ge=1, le=50)):
    """Top-N personalized recommendations from Redis cache."""
    cached = await _redis.get(f"recs:{user_id}")
    if cached:
        recs = json.loads(cached)
        results = []
        for r in recs:
            mid = r["movie_id"]
            if mid not in _movies_with_meta:
                continue                  # skip ghost IDs with no title/metadata
            results.append(
                {**get_movie_meta(mid), "score": r.get("rating", r.get("score", 0.0))}
            )
            if len(results) >= n:
                break
        if results:
            return results
    # Cold start: return popular movies
    popular = await get_popular_movies(n)
    return [{**m, "score": m.get("score", 0.0)} for m in popular]


@app.get("/similar/{movie_id}", response_model=list[SimilarMovie])
async def similar(movie_id: int, n: int = Query(10, ge=1, le=50)):
    """Item-to-item similar movies from Redis cache."""
    cached = await _redis.get(f"similar:{movie_id}")
    if cached:
        sims = json.loads(cached)[:n]
        return [
            {**get_movie_meta(s["movie_id"]), "score": s.get("rating", s.get("score", 0.0))}
            for s in sims
        ]
    raise HTTPException(status_code=404, detail="No similarity data for this movie")


@app.post("/recommend/from-likes", response_model=list[RecommendedMovie])
async def recommend_from_likes(req: LikesRequest):
    """
    Online taste-learning recommendation.

    Each call solves for the user's optimal latent-factor vector given their
    current set of liked movies (ALS update step). As more movies are liked the
    linear system becomes better conditioned and the taste vector sharpens.

    1 like  → vector points at that movie's embedding
    2 likes → vector balances both; films in the overlap rank highest
    N likes → proper taste profile; confident, genre-aware recommendations
    """
    liked_ids = {lk.movie_id for lk in req.likes}

    # Primary: ALS user-vector solve
    if _item_factors is not None:
        u = _solve_user_vector(req.likes)
        if u is not None:
            results = _recommend_from_user_vector(u, liked_ids, req.n)
            if results:
                return results

    # Fallback: Redis similarity weighted by like weight
    score_map: dict[int, float] = {}
    for lk in req.likes:
        cached = await _redis.get(f"similar:{lk.movie_id}")
        if not cached:
            continue
        for s in json.loads(cached):
            mid = int(s["movie_id"])
            if mid not in _movies_with_meta:
                continue
            score_map[mid] = (
                score_map.get(mid, 0.0) + float(s.get("score", 0.0)) * lk.weight
            )
    for mid in liked_ids:
        score_map.pop(mid, None)
    if not score_map:
        return await get_popular_movies(req.n)
    ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)[: req.n]
    return [{**get_movie_meta(mid), "score": round(sc, 4)} for mid, sc in ranked]


@app.post("/rate", status_code=202)
async def rate(req: RatingRequest, background_tasks: BackgroundTasks):
    """Ingest new user rating → publish to Kafka."""
    event = {
        "user_id": req.user_id,
        "movie_id": req.movie_id,
        "event_type": "rate",
        "rating": req.rating,
        "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
    }

    def send_to_kafka():
        if _kafka:
            _kafka.send("user-events", key=req.user_id, value=event)
            _kafka.send("rating-updates", key=req.movie_id, value=event)
        # Also invalidate the user's cached recommendations
        asyncio.run(_redis.delete(f"recs:{req.user_id}"))

    background_tasks.add_task(send_to_kafka)
    return {"status": "accepted", "event": event}


@app.get("/movies/search")
async def search_movies(q: str = Query(..., min_length=1), limit: int = Query(20, ge=1, le=100)):
    """Full-text movie search by title or genre."""
    if _movies_df is None or _movies_df.empty:
        return []
    q_lower = q.lower()
    mask = (
        _movies_df["title"].str.lower().str.contains(q_lower, na=False) |
        _movies_df["genres"].str.lower().str.contains(q_lower, na=False)
    )
    results = _movies_df[mask].sort_values("rating_count", ascending=False).head(limit)
    return [
        {
            "movie_id": int(r["movie_id"]),
            "title": str(r.get("title", "")),
            "genres": str(r.get("genres", "")),
            "avg_rating": float(r["avg_rating"]) if pd.notna(r.get("avg_rating")) else None,
            "rating_count": int(r["rating_count"]) if pd.notna(r.get("rating_count")) else None,
            "poster_url": r.get("poster_url") if pd.notna(r.get("poster_url")) else None,
        }
        for _, r in results.iterrows()
    ]


@app.get("/movies/{movie_id}")
async def movie_detail(movie_id: int):
    meta = get_movie_meta(movie_id)
    if meta["title"] == f"Movie {movie_id}":
        raise HTTPException(status_code=404, detail="Movie not found")
    # Augment with real-time stream data from Redis
    stream_feat = await _redis.hgetall(f"movie_feat:{movie_id}")
    if stream_feat:
        meta["stream_avg_rating"] = stream_feat.get("stream_avg_rating")
        meta["stream_rating_count"] = stream_feat.get("stream_rating_count")
    return meta


@app.get("/trending")
async def trending(n: int = Query(20, ge=1, le=50)):
    """Real-time trending movies from Spark Streaming."""
    cached = await _redis.get("trending:realtime")
    if cached:
        items = json.loads(cached)[:n]
        return [
            {**get_movie_meta(item["movie_id"]), "event_count": item["event_count"]}
            for item in items
        ]
    return await get_popular_movies(n)


@app.get("/popular")
async def popular(n: int = Query(20, ge=1, le=50)):
    return await get_popular_movies(n)
