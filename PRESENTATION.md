# CineRec — Large-Scale Movie Recommendation System
### SJSU CS Big Data · May 2026
**Stack:** Apache Spark · Kafka · Redis · MLflow · FastAPI · Streamlit  
**Dataset:** MovieLens 25M · 25,000,095 ratings · 162,541 users · 59,047 movies

---

---

## SLIDE 1 — Problem Statement

### What Are We Solving?

Streaming platforms like Netflix and Amazon Prime catalogue **60,000+ titles**.  
A user with no guidance will click away. **The recommendation engine is the core product.**

---

### Why Is It Hard?

| Problem | Why It Matters |
|---|---|
| **Data volume** | 25 million interactions — pandas runs out of memory |
| **Matrix sparsity** | 99.74% of the user-movie matrix is empty |
| **The long-tail** | Top 10% of movies get 86% of all ratings |
| **Cold start** | New users have no history; new movies have no interactions |
| **Latency** | Recommendations must load in < 100 ms |
| **Real-time personalisation** | User taste changes as they browse |
| **Streaming freshness** | Trending must reflect *right now*, not last week |

---

### Our Solution

```
25M historical ratings  →  Spark ETL  →  ALS training (Spark MLlib)
                                                ↓
                           Redis cache ← Precomputed recs (219K keys)
                                  ↓
Kafka live events  →  Spark Streaming  →  Redis trending feed (5-min windows)
                                  ↓
                       FastAPI (10 endpoints, < 10 ms p99)
                                  ↓
                    Streamlit UI (Netflix-style, live demo)
```

---

---

## SLIDE 2 — Datasets

### 2A · Historical Dataset — MovieLens 25M

**Source:** GroupLens Research, University of Minnesota  
**URL:** https://grouplens.org/datasets/movielens/25m/

| File | Rows | Size | Content |
|---|---|---|---|
| `ratings.csv` | 25,000,095 | 647 MB | userId, movieId, rating (0.5–5.0), timestamp |
| `movies.csv` | 62,423 | 2.9 MB | movieId, title (with year), pipe-separated genres |
| `links.csv` | 62,423 | 1.3 MB | MovieLens → IMDb → TMDB ID mapping |
| `tags.csv` | 1,093,360 | 37 MB | userId, movieId, tag text, timestamp |
| `genome-scores.csv` | 15,000,000+ | 415 MB | movieId × tagId → relevance score (0–1) |

---

### Key Statistics After ETL

| Statistic | Value |
|---|---|
| Total ratings | 25,000,095 |
| Unique users | 162,541 |
| Unique movies (with ratings) | 59,047 |
| Unique movies (with TMDB poster/metadata) | 32,722 |
| Rating scale | 0.5 to 5.0 in 0.5 steps |
| Date range | January 1995 – November 2019 |
| Median ratings per user | 71 |
| Mean ratings per user | 153.8 |
| Max ratings per user | 32,225 |
| User-item matrix sparsity | **99.74%** |

---

### Rating Distribution (25,000,095 total ratings)

```
0.5 ★  ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░   393,068   (1.6%)
1.0 ★  ████████░░░░░░░░░░░░░░░░░░░░░░░░   776,815   (3.1%)
1.5 ★  ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░   399,490   (1.6%)
2.0 ★  █████████████████░░░░░░░░░░░░░░░ 1,640,868   (6.6%)
2.5 ★  █████████████░░░░░░░░░░░░░░░░░░░ 1,262,797   (5.1%)
3.0 ★  ██████████████████████████████░░ 4,896,928  (19.6%)
3.5 ★  ██████████████████████░░░░░░░░░░ 3,177,318  (12.7%)
4.0 ★  ████████████████████████████████ 6,639,798  (26.6%) ← modal rating
4.5 ★  ██████████████████████░░░░░░░░░░ 2,200,539   (8.8%)
5.0 ★  ██████████████████████████████░░ 3,612,474  (14.4%)
```

> **Insight:** Distribution peaks at 4.0 stars — users preferentially rate movies they expected to like (selection bias). Mean = 3.53. This left-skew is why **implicit feedback mode** outperforms explicit rating prediction.

---

### 2B · Real-Time Dataset — Kafka Event Stream

**Producer:** `ingestion/kafka_producer.py` — default 50 events/sec  
**Demo mode:** `ingestion/mock_streaming.py` — writes directly to Redis, no Docker needed

**Event Schema (JSON, topic: `user-events`):**

```json
{
  "user_id":    12345,
  "movie_id":   296,
  "event_type": "rate",
  "rating":     4.5,
  "session_id": 839201,
  "timestamp":  "2026-05-01T19:30:00Z"
}
```

**Event types:** `click` · `view` · `rate` · `add_to_list`

| Kafka Topic | Producer | Consumer | Purpose |
|---|---|---|---|
| `user-events` | kafka_producer, POST /rate | spark_streaming | All user interactions |
| `rating-updates` | POST /rate | (reserved) | Rating-specific pipeline |

Spark Structured Streaming aggregates events in **5-minute tumbling windows** → top-20 trending → Redis (`trending:realtime`, TTL=5min).

---

### 2C · Enrichment — TMDB API

Raw MovieLens data has only titles and genre strings. TMDB enriched **32,044 movies** with:

- Poster URLs (`https://image.tmdb.org/t/p/w342/{poster_path}`)
- Plot overview text (1–3 sentences)
- Popularity score

**Script:** `data/tmdb_enrichment.py` → `data/enriched/tmdb_metadata.parquet` (16 MB)

---

### 2D · Data Volume Summary

| Layer | Source | Size | Records |
|---|---|---|---|
| Raw ratings (CSV) | MovieLens 25M | 647 MB | 25,000,095 |
| Raw movie metadata (CSV) | MovieLens 25M | 2.9 MB | 62,423 |
| Genome tag scores (CSV) | MovieLens 25M | 415 MB | 15M+ |
| TMDB metadata (Parquet) | TMDB API | 16 MB | 32,044 |
| Gold ratings (Parquet) | Spark ETL output | ~300 MB | 25,000,095 |
| Movie features (Parquet) | Spark ETL output | ~8 MB | 32,722 |
| User features (Parquet) | Spark ETL output | ~6 MB | 162,541 |
| ALS model artifacts | MLflow / Spark | ~171 MB | — |
| Community reviews (SQLite) | Seeded + live | ~15 MB | 147,336 |
| **Total** | | **~2.2 GB** | — |

> The 25M-record rating matrix as a dense float64 matrix would require  
> **162,541 × 59,047 × 8 bytes ≈ 76 GB** of RAM.  
> Spark's sparse representation + ALS block solver reduces this to ~10 GB driver memory.

---

---

## SLIDE 3 — System Architecture

```
┌─────────────────────────── Data Sources ───────────────────────────────┐
│                                                                          │
│  MovieLens 25M (GroupLens)    Genome Tag Scores    TMDB API (enrichment)│
│  ratings.csv   647 MB         genome-scores 415 MB  poster, overview    │
│  movies.csv    2.9 MB                               popularity score    │
│                                                                          │
└──────────────┬──────────────────────────┬───────────────────────────────┘
               │  Batch ingest             │  API calls
               ▼                           ▼
┌─────────────────────────── Ingestion Layer ────────────────────────────┐
│                                                                          │
│  Apache Kafka (Confluent 7.5.0)          Data Lake — Parquet files      │
│  ├── Topic: user-events (3 partitions)   ├── data/movielens/ml-25m/    │
│  └── Topic: rating-updates              └── data/processed/            │
│                                                                          │
└──────────────┬─────────────────────────────────────────────────────────┘
               ▼
┌─────────────────────────── Processing Layer ───────────────────────────┐
│                                                                          │
│  BATCH — Spark ETL (processing/etl_batch.py)                            │
│  Bronze → Silver → Gold (AQE + skewJoin + coalescePartitions)           │
│  ~8 minutes · 25M rows · 10 Parquet output files                        │
│                                                                          │
│  REAL-TIME — Spark Structured Streaming (ingestion/spark_streaming.py)  │
│  Query 1: 5-min tumbling window → top-20 trending → Redis               │
│  Query 2: 10-min sliding window → live avg rating → Redis               │
│  Query 3: Append raw events → Parquet bronze layer                      │
│                                                                          │
└──────────────┬─────────────────────────────────────────────────────────┘
               ▼
┌─────────────────────────── ML Layer ───────────────────────────────────┐
│                                                                          │
│  Spark MLlib ALS  (ml/train_als.py)                                      │
│  rank=100 · regParam=0.01 · maxIter=15 · alpha=25.0 · implicit=True     │
│  RMSE=0.6410 · MAE=0.4948 · MAP@10=0.0466                               │
│  Runtime: ~12 minutes local[*] · 10g driver                             │
│                                                                          │
│  Ray Tune HPO  (ml/ray_tune_als.py)                                      │
│  12 trials · ASHAScheduler · OptunaSearch (TPE)                         │
│  Best: rank=150, reg=0.05, iter=15, RMSE=0.6404                         │
│                                                                          │
│  MLflow Tracking  (mlflow/mlflow.db + mlflow/artifacts/)                │
│  14 runs · model registry · Production stage                            │
│                                                                          │
└──────────────┬─────────────────────────────────────────────────────────┘
               ▼
┌─────────────────────────── Serving Layer ──────────────────────────────┐
│                                                                          │
│  Redis 7 — 219,152 precomputed keys                                     │
│  recs:{user_id}       162,541 keys  top-20 picks per user   TTL 24h    │
│  similar:{movie_id}    56,558 keys  top-10 similar movies   TTL 24h    │
│  popular_movies             1 key   top-1000 by popularity  TTL 24h    │
│  trending:realtime          1 key   live stream top-20      TTL  5m    │
│                                                                          │
│  FastAPI + Uvicorn  (serving/api.py)  port 8000                         │
│  10 endpoints · < 10 ms p99 · SQLite reviews DB                         │
│                                                                          │
└──────────────┬─────────────────────────────────────────────────────────┘
               ▼
┌─────────────────────────── Frontend ───────────────────────────────────┐
│                                                                          │
│  Streamlit 1.33 — Netflix-style dark UI  (frontend/app.py)  port 8501  │
│  5 tabs: Home · My List · Trending · Rate · Analytics                   │
│  147K community reviews · per-profile lists · live ALS taste solve      │
│                                                                          │
└────────────────────────────────────────────────────────────────────────┘
```

---

### Technology Stack

| Layer | Technology | Version |
|---|---|---|
| Distributed processing | Apache Spark | 3.5.8 |
| ML algorithm | Spark MLlib ALS | 3.5.8 |
| Hyperparameter search | Ray Tune + Optuna | 2.20.0 |
| Experiment tracking | MLflow | 3.11.1 |
| Message broker | Apache Kafka (Confluent) | 7.5.0 |
| Stream processing | Spark Structured Streaming | 3.5.8 |
| Cache + serving | Redis | 7 |
| REST API | FastAPI + Uvicorn | 0.111.0 |
| Fuzzy search | rapidfuzz | 3.14.5 |
| Frontend | Streamlit | 1.33.0 |
| Language | Python | 3.11 |

---

---

## SLIDE 4 — Data Schema

### 4A · Parquet Data Lake (Medallion Architecture)

```
BRONZE — raw, unmodified copies
data/processed/bronze/
├── ratings/         ratings.csv → parquet, partitioned by month
├── movies/          movies.csv → parquet
└── genome_scores/   genome-scores.csv → parquet

SILVER — cleaned, joined, validated
data/processed/silver/ratings_enriched/
  user_id     INT      NOT NULL
  movie_id    INT      NOT NULL
  rating      FLOAT    ∈ [0.5, 5.0]
  timestamp   BIGINT   Unix seconds
  title       STRING
  genres      STRING   pipe-separated  e.g. "Action|Drama|Thriller"
  year        INT
  poster_url  STRING   nullable (TMDB CDN)
  overview    STRING   nullable (plot summary)

GOLD — ALS-ready rating matrix
data/processed/gold/ratings/
  user_id     INT      NOT NULL   ∈ [1, 162541]
  movie_id    INT      NOT NULL   ∈ [1, 209171]
  rating      FLOAT    NOT NULL   ∈ (0, 5.0]
  Rows: 25,000,095 · ~200 Parquet partitions
```

---

### 4B · Feature Tables

```
data/processed/movie_features/          (32,722 rows)
  movie_id      INT
  title         STRING
  genres        STRING    e.g. "Family|Comedy|Animation|Adventure"
  year          INT       nullable
  poster_url    STRING    nullable
  overview      STRING    nullable
  popularity    FLOAT     nullable (TMDB score)
  rating_count  INT
  avg_rating    FLOAT
  rating_stddev FLOAT

data/processed/user_features/           (162,541 rows)
  user_id       INT
  rating_count  INT
  avg_rating    FLOAT
  min_rating    FLOAT
  max_rating    FLOAT
```

---

### 4C · Redis Key Schema

| Key Pattern | Type | Value | TTL | Count |
|---|---|---|---|---|
| `recs:{user_id}` | JSON String | `[{movie_id, score}, ...]` top-20 | 24 h | 162,541 |
| `similar:{movie_id}` | JSON String | `[{movie_id, rating}, ...]` top-10 | 24 h | 56,558 |
| `popular_movies` | JSON String | `[{movie_id, score, ...}]` top-1000 | 24 h | 1 |
| `trending:realtime` | JSON String | `[{movie_id, event_count}]` top-20 | 5 min | 1 |
| `movie_feat:{movie_id}` | Hash | `stream_avg_rating`, `stream_rating_count` | 10 min | variable |
| `user_cache:{user_id}` | String | invalidation marker (POST /rate) | 5 min | variable |

**Total precomputed size: ~180 MB · 219,152 keys**

---

### 4D · Kafka Event Schema

```
Topic:       user-events
Partitions:  3
Replication: 1

{
  "user_id":    int,      // 1–162541
  "movie_id":   int,      // 1–209171
  "event_type": string,   // "click" | "view" | "rate" | "add_to_list"
  "rating":     float,    // 0.5–5.0, nullable for non-rate events
  "session_id": int,
  "timestamp":  string    // ISO-8601 UTC
}
```

---

### 4E · SQLite Reviews Schema

```sql
CREATE TABLE reviews (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL,
    movie_id    INTEGER NOT NULL,
    rating      REAL    NOT NULL,
    review      TEXT,
    username    TEXT,
    created_at  TEXT    NOT NULL,
    UNIQUE(user_id, movie_id)          -- one review per user per movie (upsert)
);
CREATE INDEX idx_reviews_movie ON reviews (movie_id);
```

- **147,336 pre-seeded rows** across 32,722 movies (3–6 per movie)
- Synthetic user IDs `200_001 + movie_id × 10 + i` — no collision with real users (1–162,541)
- Reviews deterministically generated: `random.Random(movie_id * 31337)` — reproducible across re-runs

---

### 4F · MLflow Tracking Schema (SQLite at `mlflow/mlflow.db`)

| Category | Fields |
|---|---|
| **Parameters** | rank, reg_param, max_iter, alpha, implicit, spark_version, dataset, mode |
| **Metrics** | rmse, mae, map_at_10, precision_at_10, coverage, train_size, test_size |
| **Artifacts** | `als-model/` (MLmodel YAML, conda.yaml, sparkml/itemFactors, sparkml/userFactors) |

**14 total runs:** 1 initial · 1 final · 12 HPO trials

---

---

## SLIDE 5 — Implementation Details

### 5A · Spark ETL Pipeline

**File:** `processing/etl_batch.py`  
**Runtime:** ~8 minutes · 10g driver · local[*]

Key Spark config:
```python
.config("spark.sql.adaptive.enabled",                    "true")   # AQE
.config("spark.sql.adaptive.coalescePartitions.enabled", "true")   # avoid 200 tiny files
.config("spark.sql.adaptive.skewJoin.enabled",           "true")   # handles Forrest Gump (81K ratings)
.config("spark.sql.shuffle.partitions",                  "100")
```

**Pipeline stages:**
```
READ       ratings.csv (25M) + movies.csv (62K) + tmdb_metadata.parquet (32K)
BRONZE     Each source → Parquet (~5× smaller than CSV)
SILVER     filter(rating ∈ [0.5,5.0]) → join(movies) → join(tmdb) → cast IDs to INT
GOLD       dropDuplicates([user_id, movie_id]) → filter(rating > 0) → 200 partitions
FEATURES   groupBy(movie_id) → avg_rating, rating_count, stddev
           groupBy(user_id)  → avg_rating, rating_count, min, max
```

> **AQE impact:** Reduced ETL runtime from ~20 min (fixed 200 partitions) to ~8 min by dynamically coalescing skewed shuffle partitions at each stage boundary.

---

### 5B · Collaborative Filtering — ALS

**File:** `ml/train_als.py`

The user-item matrix **R** (162,541 × 59,047, 99.74% sparse) is factorised into:

```
R ≈ U × Vᵀ

U ∈ ℝ^(162541 × 100)   user latent factor matrix
V ∈ ℝ^(59047  × 100)   item latent factor matrix
```

**Implicit feedback objective** (ratings = confidence, not ground truth):

```
min_{U,V}  Σ_{u,i} c_ui (p_ui − uᵀvᵢ)² + λ(||U||²_F + ||V||²_F)

p_ui = 1            if user u rated movie i  (positive preference)
c_ui = 1 + α · r_ui  confidence weight  (α=25, r_ui=star rating)
λ    = 0.01          regularisation
```

**ALS alternating step (each iteration, solved in parallel across Spark partitions):**
```
Fix V → solve for each uᵢ:  uᵢ = (VᵀCᵢV + λI)⁻¹ · Vᵀcᵢ
Fix U → solve for each vⱼ:  vⱼ = (UᵀCⱼU + λI)⁻¹ · Uᵀcⱼ
(each: a 100×100 linear system)
```

**Final model config:**
```python
ALS(rank=100, regParam=0.01, maxIter=15, alpha=25.0,
    implicitPrefs=True, coldStartStrategy="drop",
    numUserBlocks=10, numItemBlocks=10, seed=42)
```

---

### 5C · Real-Time User Vector Solve (Online Personalisation)

**Endpoint:** `POST /recommend/from-likes`  
**Latency:** < 10 ms · No model retraining

When a user adds movies to "My List", the API solves the ALS update equation on-demand:

```
X   = item_factors[liked_ids]        shape: (n_likes × 100)
c   = 1 + alpha × weights            shape: (n_likes,)
C   = diag(c)

u*  = (XᵀCX + λI)⁻¹ · Xᵀc          100×100 linear solve  < 1 ms
scores = item_factors @ u*           56K dot products       < 3 ms
```

| Liked movies | System state |
|---|---|
| 1 | Vector points directly at that movie's embedding |
| 2–3 | Balances both; overlap genres rank highest |
| 3–5 | Genre preferences emerge clearly |
| 6+ | Well-conditioned; confident, diverse recommendations |

---

### 5D · Hyperparameter Optimisation — Ray Tune

**File:** `ml/ray_tune_als.py` · 12 trials · ASHAScheduler + OptunaSearch (TPE)

| Run | Rank | Reg | Iter | RMSE | MAP@10 | Note |
|---|---|---|---|---|---|---|
| ray-tune-r150-reg0.05-i15 | 150 | 0.05 | 15 | 0.6404 | 0.0474 | Best HPO trial |
| **als-rank100-reg0.01 (chosen)** | **100** | **0.01** | **15** | **0.6410** | **0.0466** | **Final model** |
| ray-tune-r200-reg0.01-i10 | 200 | 0.01 | 10 | 0.6458 | 0.0453 | Marginal gain |
| ray-tune-r100-reg0.05-i10 | 100 | 0.05 | 10 | 0.6517 | 0.0437 | |
| ray-tune-r100-reg0.1-i10 | 100 | 0.1 | 10 | 0.6674 | 0.0414 | |
| ray-tune-r50-reg0.01-i10 | 50 | 0.01 | 10 | 0.6820 | 0.0345 | |
| ray-tune-r10-reg0.1-i10 | 10 | 0.1 | 10 | 0.7628 | 0.0155 | Underfit |

```
Effect of rank on RMSE (best reg per rank):
rank=10:  0.754  ████████████████████████████████
rank=50:  0.682  █████████████████████████████░░░
rank=100: 0.641  ███████████████████████████░░░░░
rank=150: 0.640  ███████████████████████████░░░░░  ← plateau
rank=200: 0.646  ███████████████████████████░░░░░
```

> **Why rank=100 over rank=150?** RMSE difference is 0.0006 — negligible. rank=100 trains 30% faster and uses 30% less Redis memory.

---

### 5E · Spark Structured Streaming

**File:** `ingestion/spark_streaming.py`

**Query 1 — Trending (5-minute tumbling window):**
```python
events_with_watermark = events.withWatermark("event_time", "10 minutes")
trending = (
    events_with_watermark
    .filter(col("event_type").isin("click","view","rate"))
    .groupBy(window(col("event_time"), "5 minutes"), col("movie_id"))
    .agg(count("*").alias("event_count"))
    .writeStream.foreachBatch(write_trending_to_redis)
    .trigger(processingTime="30 seconds").start()
)
```

**Query 2 — Live rating aggregates (10-min sliding, 2-min slide):**  
→ `movie_feat:{id}` hash with `stream_avg_rating` and `stream_rating_count` (TTL=10min)

**Query 3 — Bronze append (60-second micro-batches):**  
→ Raw events partitioned by `event_type` to Parquet bronze layer

---

### 5F · Serving Layer — FastAPI + Redis

**Search cascade (3 stages):**
```
Stage 1: Exact → prefix → substring  (pandas str ops, no fuzzy)
Stage 2: Genre field contains query
Stage 3: Fuzzy fallback (rapidfuzz)
         single-word: max(ratio(q, word) for word in title_words) >= 72
         multi-word:  token_sort_ratio(q, full_title) >= 65

Final score = 0.45 × fuzzy_match
            + 0.35 × (log(rating_count+1) / log(max_count+1))
            + 0.20 × (avg_rating / 5.0)
```

**API endpoints at a glance:**

| Endpoint | Mechanism | Latency |
|---|---|---|
| `GET /recommend/{user_id}` | Redis JSON get | < 5 ms |
| `POST /recommend/from-likes` | ALS 100×100 solve + 56K dot products | < 10 ms |
| `GET /similar/{movie_id}` | Redis JSON get | < 3 ms |
| `GET /movies/search?q=` | 3-stage rapidfuzz cascade | < 50 ms |
| `GET /trending` | Redis JSON get | < 5 ms |
| `GET /popular` | Redis JSON get | < 4 ms |
| `POST /rate` | Validate + Kafka + SQLite upsert | < 20 ms |
| `GET /reviews/{movie_id}` | SQLite query, newest-first | < 5 ms |

---

### 5G · Frontend — Streamlit Netflix UI

**File:** `frontend/app.py` · ~1,400 lines · 5 tabs

| Tab | What It Shows | Key Feature |
|---|---|---|
| **Home** | Daily-rotating hero + 11 genre rows + personalised row | ALS `from-likes` solve on liked movies |
| **My List** | Per-profile saved movies + taste-based picks | Independent list per profile; real-time vector solve |
| **Trending** | Live Kafka event counts + movie cards | Redis `trending:realtime` with 5-min TTL |
| **Rate** | Thumbnail search grid + star slider + review box | `POST /rate` → SQLite + Kafka + Redis invalidation |
| **Analytics** | 9 Plotly charts + dataset KPIs + MLflow metrics | Genre pie · Avg Rating by Year · Long-Tail · more |

**Movie detail panel** (click any card):
- Large poster + title + year + genres + avg rating
- TMDB plot overview
- Community reviews (2-column grid, newest first)
- Similar Movies row (7 ALS item-item neighbours)

**Community reviews system:**
- 147,336 pre-seeded reviews across 32,722 movies
- Weighted star distribution matching each movie's average rating
- Users can submit written reviews on the Rate tab — stored under their profile name

---

---

## SLIDE 6 — Visualization (LIVE DEMO)

> **Open the browser to http://localhost:8501**

---

### Demo Script — Follow This Sequence

#### Step 1 · Home Tab — Personalised Recommendations
- Select **Demo User A** (user 42) from the profile dropdown
- Point out the **hero banner** — rotates daily so it's different each day
- The **"Top Picks for You"** row loads precomputed ALS recs from Redis in < 5 ms
- Click **"Details"** on any movie → poster + overview + **community reviews** + similar movies

#### Step 2 · My List — Real-Time Taste Model
- Click **"+ List"** on 2–3 movies from different genres
- Switch to the **My List** tab
- Show the **"Because you added these"** row — it computed a new taste vector on-the-fly
- Add 2 more movies → row updates again, more refined recommendations
- Remove a movie → row instantly re-computes from remaining titles
- Switch to **Demo User B** — completely separate list, starts fresh

#### Step 3 · Trending Tab — Live Streaming
- Show the **Trending** tab with event counts and the **Live** badge
- *(if mock_streaming.py is running: counts update every 30 seconds)*
- Explain: Spark Structured Streaming aggregates Kafka events in 5-minute windows → Redis

#### Step 4 · Rate Tab — Submit a Review
- Type a movie name → **thumbnail grid** appears (not a dropdown)
- Click a thumbnail to select it
- Set star rating + type a short review
- Hit **Submit** → stored in SQLite reviews.db + published to Kafka
- Hit **"← Back"** to clear the search

#### Step 5 · Movie Details — Community Reviews
- Click **Details** on any movie
- Scroll to the **Community Reviews** section — 2-column grid with names, stars, dates
- Find your just-submitted review at the top

#### Step 6 · Analytics Tab — Data Story
- First row: **Avg Rating by Year** (survivorship bias — pre-1980 films rated higher) + **Genre Pie**
- Second row: **User Activity** + **Long-Tail** distribution
- Third row: **Movies Per Year** histogram
- Lower: **Rating Distribution** · **Genre Ratings** · **Avg Rating by Genre** · **Top N Movies**
- **Dataset KPIs**: 25M ratings · 162K users · 59K movies · 99.74% sparsity
- **Model metrics** (pulled live from MLflow SQLite): RMSE=0.6410 · MAP@10=0.0466

---

### Key Charts to Point Out

**Long-Tail Problem:**
```
Rank 1 (Forrest Gump):   81,491 ratings  ████████████████████████████████
Rank 10:                 58,773 ratings  █████████████████████░░░░░░░░░░░
Rank 100:                19,891 ratings  ████████░░░░░░░░░░░░░░░░░░░░░░░░
Rank 1,000:               2,857 ratings  █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
Rank 10,000:                 81 ratings  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░

► Top 10% of movies account for 86% of all ratings
► ALS collaborative filtering is ESSENTIAL to surface the 90% long-tail
```

**Item Similarity Validation (ALS cosine similarity — no content features used):**

| Seed Movie | Top 3 Similar |
|---|---|
| Toy Story (1995) | Toy Story 2 (0.993), Toy Story 3 (0.983), A Bug's Life (0.976) |
| The Matrix (1999) | The Matrix Reloaded (0.991), Dark City (0.964), eXistenZ (0.958) |
| Shawshank Redemption | The Green Mile (0.976), Schindler's List (0.967), The Godfather (0.959) |
| Pulp Fiction | Reservoir Dogs (0.988), Fargo (0.971), Trainspotting (0.960) |
| Star Wars: A New Hope | The Empire Strikes Back (0.994), Return of the Jedi (0.992), Raiders (0.978) |

> ALS learned **emergent semantic clusters** from interaction data alone — no genre tags or plot text were used in training.

---

---

## SLIDE 7 — Conclusion

### Summary — What Was Built End-to-End

| Component | Technology | Scale |
|---|---|---|
| Batch ETL | Spark 3.5 (AQE) | 25M rows · ~8 min |
| Collaborative filtering | Spark MLlib ALS | 162K users × 59K movies · ~12 min |
| Hyperparameter search | Ray Tune + Optuna | 12 trials · ASHAScheduler |
| Experiment tracking | MLflow | 14 runs · model registry |
| Real-time streaming | Kafka + Spark Streaming | 50 events/sec · 5-min windows |
| Serving | FastAPI + Redis | 219K precomputed keys · < 5 ms |
| Online personalisation | ALS update equation | Real-time user vector solve · < 10 ms |
| Search | rapidfuzz | 3-stage cascade on 32K titles |
| Community reviews | SQLite | 147K seeded + live user submissions |
| UI | Streamlit | Netflix-style · 5 tabs · per-profile lists |

**Final model:** RMSE=0.6410 · MAE=0.4948 · MAP@10=0.0466 · Precision@10=0.0521 · Coverage=84.7%

---

### Lessons Learned

**1. Implicit feedback is harder to evaluate than it looks.**  
RMSE = 0.64 sounds good but means little on binarised labels. MAP@10 = 0.047 is the honest number — and it reflects a genuinely hard task: recommending 10 movies out of 59,047 that a user would rate ≥ 4 stars, given only ~71 ratings as prior.

**2. Redis is the right serving layer for precomputed recs.**  
Running Spark at inference time would mean a 500 MB JVM just for serving. Redis decouples training from serving completely. The `from-likes` online solver is the escape hatch for real-time personalisation without retraining.

**3. AQE transforms Spark on uneven data.**  
The rating matrix is heavily skewed — a handful of movies have 80K+ ratings while 75% have fewer than 200. AQE's skew-join optimization and dynamic partition coalescing cut ETL from ~20 minutes to ~8 minutes.

**4. Search needs a cascade, not a single scorer.**  
Original difflib-based search ranked "W." above "Shawshank" for query "shawshnk". Moving to word-level `rfuzz.ratio` (each query word vs each title word) fixed this class of false positive entirely.

**5. Sparsity is the core challenge.**  
99.74% sparse means a nearest-neighbour system would have no neighbours for 75% of users. ALS factorises the sparse matrix into dense embeddings that generalise from observed to unobserved interactions — this is the whole point.

**6. Dynamic UI state is a first-class concern.**  
Streamlit re-renders the entire page on any state change. Dynamic tab labels (e.g. including a liked-count in the label string) caused the tabs component to reset to tab 0 on every list change. Static labels fixed this — a subtle but critical UX detail.

---

### Future Work

| Enhancement | Complexity | Impact |
|---|---|---|
| **Two-Tower neural model** (user + item encoders) | High | Better accuracy on sparse users; handles content features naturally |
| **Sequential recommendation** (SASRec / BERT4Rec) | High | Models watch-order and recent session context |
| **Multi-armed bandit for exploration** | Medium | Balances known-good recs vs. discovery for each user |
| **Real-time model updates** (online ALS) | Medium | Incorporate new ratings without full retraining; current staleness: 24h |
| **A/B testing framework** | Medium | Compare model versions on live traffic |
| **Cold-start content features** | Medium | Genre/director/cast embeddings for new movies with no ratings |
| **Diversity + fairness constraints** | High | Prevent filter bubbles; surface long-tail proportionally |
| **Kubernetes deployment** | Low | Replace Docker Compose with K8s for production ops |

---

### Thank You

**CineRec — Large-Scale Movie Recommendation System**  
SJSU CS Big Data · May 2026

> 25 million ratings · 162,541 users · 59,047 movies  
> Spark ETL → ALS → Redis → FastAPI → Streamlit  
> Live demo: http://localhost:8501

*Dataset: MovieLens 25M — GroupLens Research, University of Minnesota*  
*Model: Spark MLlib ALS (collaborative filtering, implicit feedback)*
