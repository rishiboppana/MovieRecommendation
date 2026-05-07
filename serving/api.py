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
import math
import re
import sqlite3
import difflib
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import redis.asyncio as aioredis
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable
from rapidfuzz import fuzz as rfuzz, process as rfprocess
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
REVIEWS_DB = DATA_DIR / "reviews.db"

# ─────────────────────────── Models ───────────────────────────

class RatingRequest(BaseModel):
    user_id: int = Field(..., ge=1)
    movie_id: int = Field(..., ge=1)
    rating: float = Field(..., ge=0.5, le=5.0)
    review: Optional[str] = Field(None, max_length=2000)
    username: Optional[str] = Field(None, max_length=80)

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
_titles_lower: list[str] = []            # precomputed for fast fuzzy search
_genres_lower: list[str] = []
_title_words: list[list[str]] = []       # tokenized title words for word-level fuzzy

# Precomputed search fields (kept in-memory for fast ranking)
_movies_search: pd.DataFrame | None = None

# Item-factor matrix for centroid-based taste recommendations
_item_factors: np.ndarray | None = None  # (n_movies, rank), L2-normalised
_item_ids: np.ndarray | None = None      # int32, parallel to _item_factors rows
_id_to_idx: dict[int, int] = {}          # movie_id → row index


def load_movies_df():
    """Load movie metadata once at startup for search/detail endpoints."""
    global _movies_df, _movies_with_meta, _movies_search, _titles_lower, _genres_lower, _title_words
    path = DATA_DIR / "processed" / "movie_features"
    if path.exists():
        _movies_df = pq.read_table(str(path)).to_pandas()
        _movies_with_meta = set(_movies_df["movie_id"].astype(int).tolist())
        print(f"Loaded {len(_movies_df):,} movies from feature store")
    else:
        print("WARNING: movie_features not found — run ETL first")
        _movies_df = pd.DataFrame(columns=["movie_id", "title", "genres", "avg_rating", "rating_count", "poster_url"])

    # Build a lightweight search view with pre-normalised text.
    df = _movies_df.copy()
    if df.empty:
        _movies_search = df
        return

    def _norm_text(x: object) -> str:
        s = "" if x is None else str(x)
        s = s.lower()
        s = re.sub(r"[^a-z0-9\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    df["title_norm"] = df["title"].map(_norm_text)
    df["genres_norm"] = df["genres"].map(_norm_text)
    df["title_tokens"] = df["title_norm"].map(lambda s: set(s.split()) if s else set())
    df["genres_tokens"] = df["genres_norm"].map(lambda s: set(s.split()) if s else set())
    _movies_search = df

    # Precompute lowercase lists for rapidfuzz (fast C-level fuzzy fallback)
    _titles_lower = df["title_norm"].tolist()
    _genres_lower = df["genres_norm"].tolist()
    # Word lists: filter stop words and single chars for word-level fuzzy scoring
    _stop = {"the", "a", "an", "of", "in", "and", "to", "is", "it"}
    _title_words = [
        [w for w in t.split() if len(w) >= 2 and w not in _stop] or t.split()[:1]
        for t in _titles_lower
    ]
    print(f"Search index ready: {len(_titles_lower):,} titles")


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


def init_reviews_db():
    REVIEWS_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(REVIEWS_DB))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            movie_id    INTEGER NOT NULL,
            rating      REAL    NOT NULL,
            review      TEXT,
            username    TEXT,
            created_at  TEXT    NOT NULL,
            UNIQUE(user_id, movie_id)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_reviews_movie ON reviews (movie_id)")
    # Migrate existing DBs that lack the username column
    try:
        conn.execute("ALTER TABLE reviews ADD COLUMN username TEXT")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()
    print("Reviews DB ready:", REVIEWS_DB)


def save_review(
    user_id: int,
    movie_id: int,
    rating: float,
    review: Optional[str],
    username: Optional[str] = None,
):
    conn = sqlite3.connect(str(REVIEWS_DB))
    conn.execute("""
        INSERT INTO reviews (user_id, movie_id, rating, review, username, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(user_id, movie_id) DO UPDATE SET
            rating     = excluded.rating,
            review     = excluded.review,
            username   = excluded.username,
            created_at = excluded.created_at
    """, (user_id, movie_id, rating, review or None, username or None,
          datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()


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
    init_reviews_db()
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
        "year": int(r["year"]) if pd.notna(r.get("year")) else None,
        "poster_url": r.get("poster_url") if pd.notna(r.get("poster_url")) else None,
        "overview": str(r["overview"]) if pd.notna(r.get("overview")) else None,
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


def _parse_query(q: str) -> tuple[str, set[str]]:
    q = (q or "").strip()
    q_norm = q.lower()
    q_norm = re.sub(r"[^a-z0-9\s]", " ", q_norm)
    q_norm = re.sub(r"\s+", " ", q_norm).strip()
    tokens = set(q_norm.split()) if q_norm else set()
    # Remove ultra-common tokens that add noise
    tokens.discard("the")
    tokens.discard("a")
    tokens.discard("an")
    return q_norm, tokens


def _search_score_row(
    *,
    q_norm: str,
    q_tokens: set[str],
    title_norm: str,
    title_tokens: set[str],
    genres_norm: str,
    genres_tokens: set[str],
    avg_rating: float | None,
    rating_count: int | None,
) -> float:
    """
    Fast, dependency-free ranking score.

    Priorities (roughly):
    - Exact/prefix title matches first
    - Strong substring matches
    - Token overlap + fuzzy similarity as tie-breakers
    - Popularity only as a gentle prior (prevents obscure matches dominating)
    """
    if not q_norm:
        return 0.0

    score = 0.0

    # Title match signals
    if title_norm == q_norm:
        score += 140.0
    elif title_norm.startswith(q_norm):
        score += 110.0
    elif f" {q_norm} " in f" {title_norm} ":
        score += 90.0
    elif q_norm in title_norm:
        score += 70.0

    # Genre match signals (lower weight than title)
    if genres_norm == q_norm:
        score += 40.0
    elif genres_norm.startswith(q_norm):
        score += 25.0
    elif q_norm in genres_norm:
        score += 15.0

    # Token overlap (title)
    if q_tokens:
        inter = len(q_tokens & title_tokens)
        if inter:
            score += 12.0 * inter
        # Jaccard-ish boost
        union = len(q_tokens | title_tokens)
        if union:
            score += 35.0 * (inter / union)

    # Fuzzy similarity (typo tolerance).
    # Single-word queries: compare against each title word and take the best match.
    # This avoids partial_ratio falsely scoring short titles (e.g. "W." vs "shawshnk").
    # Multi-word queries: token_sort_ratio handles word reordering naturally.
    if len(q_norm) >= 3 and title_tokens:
        if " " not in q_norm:
            best_word = max(
                (rfuzz.ratio(q_norm, w) for w in title_tokens if len(w) >= 2),
                default=0.0,
            )
            score += 0.55 * best_word
        else:
            score += 0.55 * rfuzz.token_sort_ratio(q_norm, title_norm)

    # Gentle priors
    if rating_count is not None and rating_count > 0:
        score += 2.0 * math.log1p(float(rating_count))
    if avg_rating is not None:
        score += 1.5 * float(avg_rating)

    return float(score)


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
    """Ingest new user rating + optional review → Kafka + SQLite."""
    event = {
        "user_id": req.user_id,
        "movie_id": req.movie_id,
        "event_type": "rate",
        "rating": req.rating,
        "timestamp": datetime.utcnow().isoformat(),
    }

    save_review(req.user_id, req.movie_id, req.rating, req.review, req.username)

    def send_to_kafka():
        if _kafka:
            _kafka.send("user-events", key=req.user_id, value=event)
            _kafka.send("rating-updates", key=req.movie_id, value=event)
        asyncio.run(_redis.delete(f"recs:{req.user_id}"))

    background_tasks.add_task(send_to_kafka)
    return {"status": "accepted", "event": event}


@app.get("/reviews/{movie_id}")
async def get_reviews(
    movie_id: int,
    limit: int = Query(20, ge=1, le=100),
):
    """Return user reviews for a movie (most recent first)."""
    if not REVIEWS_DB.exists():
        return []
    conn = sqlite3.connect(str(REVIEWS_DB))
    rows = conn.execute("""
        SELECT user_id, rating, review, username, created_at
        FROM reviews
        WHERE movie_id = ?
        ORDER BY created_at DESC
        LIMIT ?
    """, (movie_id, limit)).fetchall()
    conn.close()
    return [
        {
            "user_id": r[0],
            "rating": r[1],
            "review": r[2],
            "username": r[3],
            "created_at": r[4],
        }
        for r in rows
    ]


@app.get("/movies/search")
async def search_movies(q: str = Query(..., min_length=1), limit: int = Query(20, ge=1, le=100)):
    """Ranked movie search by title/genre (typo-tolerant, title-first)."""
    if _movies_search is None or _movies_search.empty:
        return []
    q_norm, q_tokens = _parse_query(q)
    if not q_norm:
        return []

    df = _movies_search

    # Candidate selection: prefer cheap containment before scoring everything.
    # Also allow token-level OR matching for short queries.
    if len(q_norm) <= 3 and q_tokens:
        # For very short queries, use token overlap to avoid returning nothing.
        mask = df["title_tokens"].map(lambda ts: bool(ts & q_tokens)) | df["genres_tokens"].map(lambda ts: bool(ts & q_tokens))
    else:
        mask = df["title_norm"].str.contains(q_norm, na=False) | df["genres_norm"].str.contains(q_norm, na=False)

    cands = df[mask]
    if cands.empty and len(q_norm) >= 3 and _title_words:
        # Typo-only query: word-level ratio avoids spurious matches on short titles.
        if " " not in q_norm:
            # Single-word query: find titles where any word scores ≥ 72 against the query.
            scored_idx = [
                (max((rfuzz.ratio(q_norm, w) for w in words if len(w) >= 2), default=0), i)
                for i, words in enumerate(_title_words)
            ]
            fuzzy_idx = [i for s, i in scored_idx if s >= 72]
        else:
            # Multi-word query: token_sort_ratio across all titles (still fast enough)
            fuzzy_hits = rfprocess.extract(
                q_norm, _titles_lower, scorer=rfuzz.token_sort_ratio,
                limit=min(limit * 4, 200), score_cutoff=65,
            )
            fuzzy_idx = [h[2] for h in fuzzy_hits]

        if fuzzy_idx:
            cands = df.iloc[fuzzy_idx]
        else:
            cands = df.sort_values("rating_count", ascending=False).head(500)

    # Score and rank
    scored = []
    for _, r in cands.iterrows():
        avg_rating = float(r["avg_rating"]) if pd.notna(r.get("avg_rating")) else None
        rating_count = int(r["rating_count"]) if pd.notna(r.get("rating_count")) else None
        s = _search_score_row(
            q_norm=q_norm,
            q_tokens=q_tokens,
            title_norm=str(r.get("title_norm") or ""),
            title_tokens=r.get("title_tokens") or set(),
            genres_norm=str(r.get("genres_norm") or ""),
            genres_tokens=r.get("genres_tokens") or set(),
            avg_rating=avg_rating,
            rating_count=rating_count,
        )
        if s <= 0:
            continue
        scored.append((s, r))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:limit]

    out = []
    for s, r in top:
        out.append(
            {
                "movie_id": int(r["movie_id"]),
                "title": str(r.get("title", "")),
                "genres": str(r.get("genres", "")),
                "avg_rating": float(r["avg_rating"]) if pd.notna(r.get("avg_rating")) else None,
                "rating_count": int(r["rating_count"]) if pd.notna(r.get("rating_count")) else None,
                "poster_url": r.get("poster_url") if pd.notna(r.get("poster_url")) else None,
                # Debug-friendly ranking signal (higher = better match).
                "match_score": round(float(s), 4),
            }
        )
    return out


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
