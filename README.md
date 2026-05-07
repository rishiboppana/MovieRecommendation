# CineRec — Large-Scale Movie Recommendation System

**SJSU CS Big Data · Semester Project**  
**Dataset:** MovieLens 25M · 25,000,095 ratings · 162,541 users · 59,047 movies  
**Stack:** Apache Spark · Kafka · Redis · MLflow · FastAPI · Streamlit

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Datasets](#2-datasets)
   - 2.1 [Historical Dataset — MovieLens 25M](#21-historical-dataset--movielens-25m)
   - 2.2 [Real-Time Dataset — Kafka Event Stream](#22-real-time-dataset--kafka-event-stream)
   - 2.3 [Enrichment — TMDB API](#23-enrichment--tmdb-api)
   - 2.4 [Data Volume Summary](#24-data-volume-summary)
3. [System Architecture](#3-system-architecture)
4. [Data Schema](#4-data-schema)
   - 4.1 [Parquet Data Lake (Bronze / Silver / Gold)](#41-parquet-data-lake-bronze--silver--gold)
   - 4.2 [Redis Key Schema](#42-redis-key-schema)
   - 4.3 [Kafka Event Schema](#43-kafka-event-schema)
   - 4.4 [MLflow Tracking Schema](#44-mlflow-tracking-schema)
5. [Implementation Details](#5-implementation-details)
   - 5.1 [Spark ETL Pipeline](#51-spark-etl-pipeline)
   - 5.2 [Collaborative Filtering with ALS](#52-collaborative-filtering-with-als)
   - 5.3 [Real-Time User Vector Solve](#53-real-time-user-vector-solve)
   - 5.4 [Hyperparameter Optimisation with Ray Tune](#54-hyperparameter-optimisation-with-ray-tune)
   - 5.5 [Spark Structured Streaming](#55-spark-structured-streaming)
   - 5.6 [Serving Layer — FastAPI + Redis](#56-serving-layer--fastapi--redis)
   - 5.7 [Frontend — Streamlit Netflix UI](#57-frontend--streamlit-netflix-ui)
6. [Live Demo](#6-live-demo)
7. [Results & Visualizations](#7-results--visualizations)
   - 7.1 [Dataset Analytics](#71-dataset-analytics)
   - 7.2 [Model Performance](#72-model-performance)
   - 7.3 [Hyperparameter Sweep Results](#73-hyperparameter-sweep-results)
   - 7.4 [System Latency](#74-system-latency)
   - 7.5 [Item Similarity Validation](#75-item-similarity-validation)
8. [Running the Application](#8-running-the-application)
9. [API Reference](#9-api-reference)
10. [Project Structure](#10-project-structure)
11. [Conclusion](#11-conclusion)

---

## 1. Problem Statement

### The Challenge

Streaming platforms like Netflix, Amazon Prime, and Disney+ each catalogue tens of thousands of titles. A user confronted with 60,000+ movies and no guidance will click away. The recommendation engine is the core product, not the catalogue itself.

Building a recommendation system that works at scale introduces several hard technical problems:

| Problem | Why It Is Hard |
|---|---|
| **Data volume** | 25 million rating interactions require distributed processing — pandas runs out of memory |
| **Matrix sparsity** | The user-movie rating matrix is 99.74% empty; most users have never seen most movies |
| **The long-tail problem** | 10% of movies account for 86% of all ratings; the other 90% of the catalogue is nearly invisible |
| **Cold start** | New users have no rating history; new movies have no interaction data |
| **Latency** | A recommendation page must load in < 100 ms; batch model inference is too slow |
| **Real-time personalisation** | User taste changes as they browse; the model must adapt without full retraining |
| **Streaming freshness** | Trending content must reflect what people are watching *right now*, not last week |

### Our Solution

CineRec addresses all of these by building a complete production-grade recommendation pipeline:

```
25M historical ratings   →  Spark ETL  →  ALS training (Spark MLlib)
                                                 ↓
                              Redis cache ← Precomputed recs (219 K keys)
                                   ↓
Kafka live events  →  Spark Streaming  →  Redis trending feed (5-min windows)
                                   ↓
                        FastAPI (8 endpoints, < 10 ms p99)
                                   ↓
                     Streamlit UI (Netflix-style, live demo)
```

The system demonstrates:
- **Distributed batch processing** of 25M records at scale with Spark
- **Collaborative filtering** via Spark MLlib ALS with implicit feedback
- **Real-time event processing** via Kafka + Spark Structured Streaming
- **Online personalisation** by solving the ALS update equation on-demand (no retraining)
- **Full MLOps loop** with MLflow experiment tracking, model registry, and HPO

---

## 2. Datasets

### 2.1 Historical Dataset — MovieLens 25M

The primary dataset is **MovieLens 25M**, released by GroupLens Research at the University of Minnesota. It is the gold standard for evaluating collaborative filtering systems.

**Source:** https://grouplens.org/datasets/movielens/25m/

| File | Rows | Size | Content |
|---|---|---|---|
| `ratings.csv` | 25,000,095 | 647 MB | userId, movieId, rating (0.5–5.0), timestamp |
| `movies.csv` | 62,423 | 2.9 MB | movieId, title (with year), pipe-separated genres |
| `links.csv` | 62,423 | 1.3 MB | MovieLens → IMDb → TMDB ID mapping |
| `tags.csv` | 1,093,360 | 37 MB | userId, movieId, tag text, timestamp |
| `genome-scores.csv` | 15,000,000+ | 415 MB | movieId × tagId → relevance score (0–1) |
| `genome-tags.csv` | 13,176 | 18 KB | tagId → tag label vocabulary |

**Dataset statistics after ETL:**

| Statistic | Value |
|---|---|
| Total ratings | 25,000,095 |
| Unique users | 162,541 |
| Unique movies (with ratings) | 59,047 |
| Unique movies (with TMDB metadata) | 32,722 |
| Rating scale | 0.5 to 5.0 in 0.5 steps |
| Date range | January 1995 – November 2019 |
| Median ratings per user | 71 |
| Mean ratings per user | 153.8 |
| Max ratings per user | 32,225 |
| Mean ratings per movie | 762 |
| Max ratings per movie | 81,491 (Forrest Gump) |
| User-item matrix sparsity | **99.74%** |

**Rating distribution across all 25 million ratings:**

```
0.5 ★  ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░   393,068   (1.6%)
1.0 ★  ████████░░░░░░░░░░░░░░░░░░░░░░░░   776,815   (3.1%)
1.5 ★  ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░   399,490   (1.6%)
2.0 ★  █████████████████░░░░░░░░░░░░░░░ 1,640,868   (6.6%)
2.5 ★  █████████████░░░░░░░░░░░░░░░░░░░ 1,262,797   (5.1%)
3.0 ★  ██████████████████████████████░░ 4,896,928  (19.6%)
3.5 ★  ██████████████████████░░░░░░░░░░ 3,177,318  (12.7%)
4.0 ★  ████████████████████████████████ 6,639,798  (26.6%) ← most common
4.5 ★  ██████████████████████░░░░░░░░░░ 2,200,539   (8.8%)
5.0 ★  ██████████████████████████████░░ 3,612,474  (14.4%)
```

*Insight: The distribution is left-skewed (mean 3.53). Users preferentially rate movies they chose to watch and expected to like — selection bias means 4 stars is the modal rating.*

### 2.2 Real-Time Dataset — Kafka Event Stream

In addition to the static historical dataset, CineRec ingests a live event stream simulating user interactions on the platform.

**Event producer:** `ingestion/kafka_producer.py` (configurable rate, default 50 events/sec)  
**Demo mode:** `ingestion/mock_streaming.py` (no Kafka/Docker required — writes directly to Redis)

**Event schema (JSON, published to topic `user-events`):**

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

Event types: `click`, `view`, `rate`, `add_to_list`

**Kafka topics:**

| Topic | Producer | Consumer | Purpose |
|---|---|---|---|
| `user-events` | kafka_producer, POST /rate | spark_streaming | All user interactions |
| `rating-updates` | POST /rate | (reserved) | Rating-specific pipeline |

The Spark Structured Streaming consumer aggregates events in 5-minute tumbling windows and writes the top-20 trending movies to Redis with a 5-minute TTL. The Trending tab in the UI shows this live data.

### 2.3 Enrichment — TMDB API

The raw MovieLens data has titles and pipe-separated genre strings but no poster images or plot summaries. The TMDB (The Movie Database) API was used to enrich 32,044 movies with:

- Poster URLs (`https://image.tmdb.org/t/p/w342/{poster_path}`)
- Plot overview text (1–3 sentences)
- Popularity score (TMDB's proprietary engagement metric)

**Enrichment script:** `data/tmdb_enrichment.py`  
**Output:** `data/enriched/tmdb_metadata.parquet` (16 MB)

To replicate enrichment:
```bash
export TMDB_API_KEY=your_key_here
python3 data/tmdb_enrichment.py --limit 500   # quick demo: top 500 movies
python3 data/tmdb_enrichment.py               # full: all 62K movies (~1 hour)
```

### 2.4 Data Volume Summary

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
| **Total** | | **~2.2 GB** | — |

The 25M-record rating matrix, when stored as a dense float64 matrix, would require **162,541 × 59,047 × 8 bytes ≈ 76 GB** of RAM. Spark's sparse matrix representation and ALS's block-partitioned solver reduce this to ~10 GB driver memory.

---

## 3. System Architecture

```
┌─────────────────────────────────── Data Sources ────────────────────────────────────┐
│                                                                                       │
│   MovieLens 25M (GroupLens)          Genome Tag Scores        TMDB API (enrichment)  │
│   ratings.csv   647 MB               genome-scores.csv 415 MB  poster_url, overview  │
│   movies.csv    2.9 MB               genome-tags.csv   18 KB   popularity score      │
│                                                                                       │
└──────────────────┬───────────────────────────┬──────────────────────────────────────┘
                   │  Batch ingest              │  API calls
                   ▼                            ▼
┌─────────────────────────────────── Ingestion Layer ─────────────────────────────────┐
│                                                                                       │
│   Apache Kafka (Confluent 7.5.0)                  Data Lake — Parquet files          │
│   ├── Broker: localhost:29092                      ├── data/movielens/ml-25m/        │
│   ├── Topic: user-events (3 partitions)            ├── data/enriched/                │
│   └── Topic: rating-updates                        └── data/processed/              │
│                                                                                       │
│   kafka_producer.py  ──►  user-events              ingestion/init_topics.py          │
│   POST /rate         ──►  user-events              (creates topics on startup)       │
│                                                                                       │
└────────────────────┬─────────────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────── Processing Layer ────────────────────────────────┐
│                                                                                       │
│   BATCH — Spark ETL (processing/etl_batch.py)                                        │
│   ┌─────────────────────────────────────────────────────────────────┐                │
│   │  Bronze  Raw parquet from CSV sources                           │                │
│   │  Silver  Cleaned + joined (ratings ⋈ movies ⋈ TMDB metadata)   │                │
│   │  Gold    ALS-ready: deduplicated, integer IDs, non-null         │                │
│   └─────────────────────────────────────────────────────────────────┘                │
│   Config: AQE=true, coalescePartitions=true, skewJoin=true, shuffle_parts=100        │
│   Runtime: ~8 minutes (local[*], 8g driver, 25M rows)                               │
│                                                                                       │
│   FEATURE ENGINEERING (processing/feature_engineering.py)                            │
│   └── Genre one-hot (19 genres) · Rating bias decomposition · Implicit confidence    │
│                                                                                       │
│   REAL-TIME — Spark Structured Streaming (ingestion/spark_streaming.py)              │
│   ┌─────────────────────────────────────────────────────────────────┐                │
│   │  Query 1  5-min tumbling window → top-20 trending → Redis       │                │
│   │  Query 2  10-min sliding window → live avg rating → Redis       │                │
│   │  Query 3  Append raw events → Parquet bronze layer              │                │
│   └─────────────────────────────────────────────────────────────────┘                │
│   Trigger: processingTime="30 seconds"                                               │
│                                                                                       │
└────────────────────┬─────────────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────── ML Layer ────────────────────────────────────────┐
│                                                                                       │
│   Spark MLlib ALS  (ml/train_als.py)                                                  │
│   ├── Mode: implicitPrefs=True  (confidence weighting: c = 1 + 25 × rating)          │
│   ├── rank=100 · regParam=0.01 · maxIter=15 · alpha=25.0                             │
│   ├── Output: U ∈ ℝ^(162541×100)  V ∈ ℝ^(59047×100)                                 │
│   ├── RMSE=0.6410  MAE=0.4948  MAP@10=0.0466  Precision@10=0.0521                   │
│   └── Runtime: ~12 minutes on local[*] with 10g driver                               │
│                                                                                       │
│   Ray Tune HPO  (ml/ray_tune_als.py)                                                  │
│   ├── Scheduler: ASHAScheduler (early stopping)                                      │
│   ├── Sampler: OptunaSearch (TPE)                                                    │
│   ├── Search space: rank ∈ {10,50,100,150,200}, reg ∈ {0.01..0.5}, iter ∈ {5..15}  │
│   └── 12 trials completed → best: rank=150, reg=0.05, iter=15 (RMSE=0.6404)         │
│                                                                                       │
│   MLflow Tracking  (mlflow/mlflow.db · mlflow/artifacts/)                             │
│   ├── Experiment: movie-recommendations-als                                           │
│   ├── 14 runs total (1 initial, 1 final, 12 HPO trials)                              │
│   └── Model Registry: movie-rec-als v1 → stage=Production                           │
│                                                                                       │
└────────────────────┬─────────────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────── Serving Layer ───────────────────────────────────┐
│                                                                                       │
│   Redis 7 (219,152 precomputed keys)                                                  │
│   ├── recs:{user_id}        162,541 keys — top-20 picks per user    TTL 24h          │
│   ├── similar:{movie_id}     56,558 keys — top-10 similar movies    TTL 24h          │
│   ├── popular_movies               1 key  — top-1000 by popularity  TTL 24h          │
│   └── trending:realtime            1 key  — live stream top-20      TTL  5m          │
│                                                                                       │
│   FastAPI + Uvicorn  (serving/api.py)  port 8000                                      │
│   ├── In-memory: movies_df (32,722 × 10)  item_factors (56,558 × 100)               │
│   ├── GET /recommend/{user_id}           < 5 ms   (Redis hit)                        │
│   ├── POST /recommend/from-likes         < 10 ms  (ALS user-vector solve)            │
│   ├── GET /similar/{movie_id}            < 3 ms   (Redis hit)                        │
│   ├── GET /movies/search?q=...           < 50 ms  (rapidfuzz on pandas)              │
│   ├── GET /movies/{movie_id}             < 5 ms   (metadata lookup)                  │
│   ├── GET /trending                      < 5 ms   (Redis hit)                        │
│   ├── GET /popular                       < 5 ms   (Redis hit)                        │
│   ├── POST /rate                         < 20 ms  (validate + Kafka + SQLite save)   │
│   └── GET /reviews/{movie_id}            < 5 ms   (SQLite query, newest-first)       │
│                                                                                       │
└────────────────────┬─────────────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────── Frontend ────────────────────────────────────────┐
│                                                                                       │
│   Streamlit 1.33 — Netflix-style dark UI  (frontend/app.py)  port 8501               │
│   ├── Home      Daily-rotating hero + genre rows + personalised picks (ALS)          │
│   ├── My List   Per-profile saved movies → real-time ALS taste solve                 │
│   ├── Trending  Live Kafka/Spark streaming results with event counts                 │
│   ├── Rate      Thumbnail search + rating + written review → SQLite + Kafka          │
│   └── Analytics Dataset KPIs + 9 Plotly charts (pie, scatter, bar) + MLflow metrics │
│                                                                                       │
│   Movie detail panel (poster + overview + community reviews + similar movies row)    │
│                                                                                       │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

**Technology stack:**

| Layer | Technology | Version |
|---|---|---|
| Distributed processing | Apache Spark | 3.5.8 |
| ML algorithm | Spark MLlib ALS | 3.5.8 |
| Hyperparameter search | Ray Tune + Optuna | 2.20.0 |
| Experiment tracking | MLflow | 3.11.1 |
| Message broker | Apache Kafka (Confluent) | 7.5.0 |
| Stream processing | Spark Structured Streaming | 3.5.8 |
| Cache + serving | Redis | 7 / 8.6.2 |
| REST API | FastAPI + Uvicorn | 0.111.0 |
| Fuzzy search | rapidfuzz | 3.14.5 |
| Frontend | Streamlit | 1.33.0 |
| Orchestration | Docker Compose | 27.5.1 |
| Language | Python | 3.11 |

---

## 4. Data Schema

### 4.1 Parquet Data Lake (Bronze / Silver / Gold)

The ETL pipeline follows the **medallion architecture** with three logical layers written as Parquet files.

**Bronze layer** — raw, unmodified copies of source data:
```
data/processed/bronze/
├── ratings/          # ratings.csv → parquet, partitioned by month
├── movies/           # movies.csv → parquet
└── genome_scores/    # genome-scores.csv → parquet
```

**Silver layer** — cleaned, joined, validated:
```
data/processed/silver/
└── ratings_enriched/
    # Schema:
    # user_id      INT      NOT NULL
    # movie_id     INT      NOT NULL
    # rating       FLOAT    ∈ [0.5, 5.0]
    # timestamp    BIGINT   Unix seconds
    # title        STRING
    # genres       STRING   pipe-separated, e.g. "Action|Drama|Thriller"
    # year         INT
    # poster_url   STRING   nullable
    # overview     STRING   nullable
```

**Gold layer** — ALS-ready rating matrix:
```
data/processed/gold/ratings/
    # Schema:
    # user_id      INT      NOT NULL  ∈ [1, 162541]
    # movie_id     INT      NOT NULL  ∈ [1, 209171]
    # rating       FLOAT    NOT NULL  ∈ (0, 5.0]
    # Rows: 25,000,095
    # Partitions: ~200 Parquet parts
```

**Feature tables** (computed by ETL, loaded by API at startup):

```
data/processed/movie_features/
    # movie_id      INT
    # title         STRING
    # genres        STRING   e.g. "Family|Comedy|Animation|Adventure"
    # year          INT      nullable
    # poster_url    STRING   nullable (TMDB CDN URL)
    # overview      STRING   nullable (plot summary)
    # popularity    FLOAT    nullable (TMDB popularity score)
    # rating_count  INT      number of ratings in MovieLens
    # avg_rating    FLOAT    mean star rating
    # rating_stddev FLOAT    standard deviation of ratings
    # Rows: 32,722

data/processed/user_features/
    # user_id       INT
    # rating_count  INT
    # avg_rating    FLOAT
    # min_rating    FLOAT
    # max_rating    FLOAT
    # Rows: 162,541
```

**ALS model artifacts** (saved by Spark MLlib):
```
mlflow/artifacts/als-model-local/
    ├── itemFactors/   # 56,558 × 100 dense float32 matrix (item latent vectors)
    └── userFactors/   # 162,541 × 100 dense float32 matrix (user latent vectors)
```

### 4.2 Redis Key Schema

All 219,152 keys follow these naming conventions:

| Key Pattern | Type | Value | TTL | Count |
|---|---|---|---|---|
| `recs:{user_id}` | String (JSON) | `[{"movie_id":int, "score":float}, ...]` top-20 | 24 h | 162,541 |
| `similar:{movie_id}` | String (JSON) | `[{"movie_id":int, "rating":float}, ...]` top-10 | 24 h | 56,558 |
| `popular_movies` | String (JSON) | `[{"movie_id":int, "score":float, ...}, ...]` top-1000 | 24 h | 1 |
| `trending:realtime` | String (JSON) | `[{"movie_id":int, "event_count":int}, ...]` top-20 | 5 min | 1 |
| `movie_feat:{movie_id}` | Hash | `stream_avg_rating`, `stream_rating_count` | 10 min | variable |
| `user_cache:{user_id}` | String | invalidation marker (written by POST /rate) | 5 min | variable |

**Total precomputed size:** ~180 MB in Redis memory

### 4.3 Kafka Event Schema

```
Topic: user-events
Partitions: 3
Replication: 1

Message format: JSON (UTF-8)

{
  "user_id":    int,      // 1–162541
  "movie_id":   int,      // 1–209171
  "event_type": string,   // "click" | "view" | "rate" | "add_to_list"
  "rating":     float,    // 0.5–5.0, nullable for non-rate events
  "session_id": int,      // random session identifier
  "timestamp":  string    // ISO-8601 UTC
}

Example:
{
  "user_id": 12345, "movie_id": 296, "event_type": "rate",
  "rating": 4.5, "session_id": 839201, "timestamp": "2026-05-01T19:30:00Z"
}
```

### 4.4 MLflow Tracking Schema

MLflow uses a SQLite backend at `mlflow/mlflow.db`.

**Logged per training run:**

| Category | Fields |
|---|---|
| **Parameters** | rank, reg_param, max_iter, alpha, implicit, spark_version, dataset, mode, num_users, num_movies, total_ratings |
| **Metrics** | rmse, mae, map_at_10, precision_at_10, coverage, train_size, test_size |
| **Tags** | model_registered, model_stage, mlflow.runName, mlflow.source.type |
| **Artifacts** | `als-model/` (MLmodel YAML, conda.yaml, sparkml/itemFactors, sparkml/userFactors) |

---

## 5. Implementation Details

### 5.1 Spark ETL Pipeline

**File:** `processing/etl_batch.py`

The ETL pipeline processes 25M records using Spark's **Adaptive Query Execution (AQE)**, which dynamically re-optimises the physical plan at each shuffle boundary. Key configuration:

```python
SparkSession.builder
    .config("spark.sql.adaptive.enabled",                    "true")
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true")  # avoid 200 tiny files
    .config("spark.sql.adaptive.skewJoin.enabled",           "true")  # handles popular-movie skew
    .config("spark.sql.shuffle.partitions",                  "100")
    .config("spark.driver.memory",                           "8g")
```

**Pipeline stages:**

```
1. READ
   ratings.csv     → spark.read.csv(inferSchema=True)  → 25,000,095 rows
   movies.csv      → spark.read.csv(inferSchema=True)  → 62,423 rows
   tmdb_metadata   → spark.read.parquet()              → 32,044 rows

2. BRONZE
   Each source written to Parquet (columnar, ~5× smaller than CSV).

3. SILVER
   ratings
     .filter(col("rating").between(0.5, 5.0))          # removes invalid ratings
     .filter(col("userId").isNotNull())
     .join(movies, "movieId", "left")                   # add title, genres
     .join(tmdb,   "movieId", "left")                   # add poster_url, overview
     .withColumn("user_id",  col("userId").cast(IntegerType()))
     .withColumn("movie_id", col("movieId").cast(IntegerType()))
   → 25,000,095 rows (no drop — all valid)

4. GOLD
   silver
     .select("user_id", "movie_id", "rating")
     .dropDuplicates(["user_id", "movie_id"])           # deduplicate (keep first rating)
     .filter(col("rating") > 0)
   → Saved as Parquet with 200 partitions for balanced ALS block reads.

5. FEATURE TABLES
   movie_features: groupBy(movie_id) → avg_rating, rating_count, stddev
   user_features:  groupBy(user_id)  → avg_rating, rating_count, min, max
```

**Feature engineering** (`processing/feature_engineering.py`):

```python
# Genre one-hot encoding (19 genres → 19 binary columns)
for genre in ALL_GENRES:
    df = df.withColumn(f"genre_{genre}", col("genres").contains(genre).cast(IntegerType()))

# Rating bias decomposition
global_mean = ratings_df.agg(avg("rating")).first()[0]
user_bias   = ratings_df.groupBy("user_id").agg((avg("rating") - global_mean).alias("user_bias"))
item_bias   = ratings_df.groupBy("movie_id").agg((avg("rating") - global_mean).alias("item_bias"))

# Implicit confidence weights
df = df.withColumn("confidence", lit(1.0) + lit(alpha) * col("rating"))  # c = 1 + 25r
```

### 5.2 Collaborative Filtering with ALS

**File:** `ml/train_als.py`

**Matrix factorisation objective (implicit feedback mode):**

The user-item matrix **R** (162,541 × 59,047, 99.74% sparse) is factored into:

```
R ≈ U × Vᵀ

U ∈ ℝ^(n_users × rank)   — user latent factor matrix   (162,541 × 100)
V ∈ ℝ^(n_items × rank)   — item latent factor matrix   (59,047  × 100)
```

The objective function under implicit feedback treats ratings as confidence signals, not ground-truth scores:

```
min_{U,V}  Σ_{u,i} c_ui (p_ui − uᵀvᵢ)² + λ(||U||²_F + ||V||²_F)

where:
  p_ui = 1       if user u rated movie i (positive preference)
  c_ui = 1 + α·r_ui  confidence weight  (α=25, r_ui=rating value)
  λ    = 0.01    regularisation strength
```

**Why implicit mode?** Explicit ALS predicts the exact star rating. Implicit ALS treats any interaction (even a mediocre 2-star rating) as evidence the user *watched and engaged with* a movie — which is a better signal for "what will this user watch next?" than "what will this user rate 5 stars?"

**ALS alternating step (one iteration):**

```
Fix V, solve for each row uᵢ of U in closed form:
  uᵢ = (VᵀCᵢV + λI)⁻¹ · Vᵀcᵢ
  (a 100×100 linear system, parallelised across Spark partitions)

Fix U, solve for each row vⱼ of V:
  vⱼ = (UᵀCⱼU + λI)⁻¹ · Uᵀcⱼ
```

**Spark MLlib ALS configuration:**

```python
ALS(
    rank            = 100,    # dimensionality of latent vectors
    regParam        = 0.01,   # L2 regularisation
    maxIter         = 15,     # alternating steps
    alpha           = 25.0,   # confidence scaling
    implicitPrefs   = True,   # implicit feedback objective
    userCol         = "user_id",
    itemCol         = "movie_id",
    ratingCol       = "rating",
    coldStartStrategy = "drop",  # exclude unknown users/items from eval
    numUserBlocks   = 10,     # partition U into 10 blocks
    numItemBlocks   = 10,     # partition V into 10 blocks
    seed            = 42,
)
```

**Training / test split:**

```python
train_df, test_df = ratings.randomSplit([0.8, 0.2], seed=42)
# train: ~20,000,000 ratings
# test:   ~5,000,000 ratings
```

**Evaluation** on the test split with binarised labels (rating ≥ 4.0 = positive):

```python
predictions = model.transform(test_df).withColumn(
    "binary_label",
    when(col("rating") >= 4.0, 1.0).otherwise(0.0)
)
evaluator_rmse = RegressionEvaluator(metricName="rmse", labelCol="binary_label", ...)
evaluator_mae  = RegressionEvaluator(metricName="mae",  labelCol="binary_label", ...)
```

### 5.3 Real-Time User Vector Solve

**Endpoint:** `POST /recommend/from-likes`

When a user adds movies to "My List", the API solves for their optimal latent vector using the ALS update equation — no model retraining required. This is the key innovation enabling real-time personalisation:

```
Given: liked movie IDs with confidence weights

X   = item_factor_matrix[liked_ids]    shape: (n_likes × 100)
C   = diag(1 + alpha × weights)        shape: (n_likes × n_likes)
c   = (1 + alpha × weights)            shape: (n_likes,)

Solve: u* = (XᵀCX + λI)⁻¹ · Xᵀc     shape: (100,)
       (100×100 system, < 1 ms)

Score all movies:
       scores = item_factor_matrix @ u*  shape: (56,558,)
       (single matrix-vector multiply, < 3 ms)
```

**Python implementation:**

```python
X  = item_factors[liked_indices]                   # (n × 100)
c  = 1.0 + alpha * np.array(weights)               # confidence vector
C  = np.diag(c)
A  = X.T @ C @ X + lam * np.eye(rank)              # (100×100)
b  = X.T @ c                                       # (100,)
u_star = np.linalg.solve(A, b)                     # (100,)
scores = item_factors @ u_star                     # (56558,)
```

**Taste model quality:**

| Liked movies | System state |
|---|---|
| 1 | User vector points directly at that movie's embedding |
| 2 | Vector balances both; films in their overlap rank highest |
| 3–5 | System becomes better conditioned; genre preferences emerge |
| 6+ | Well-conditioned linear system; confident, diverse recommendations |

### 5.4 Hyperparameter Optimisation with Ray Tune

**File:** `ml/ray_tune_als.py`

12 HPO trials were run using **Ray Tune** with **ASHAScheduler** (early stopping) and **OptunaSearch** (Tree-structured Parzen Estimator):

```python
search_space = {
    "rank":      tune.choice([10, 50, 100, 150, 200]),
    "reg_param": tune.loguniform(0.001, 0.5),
    "max_iter":  tune.choice([5, 10, 15]),
    "alpha":     tune.uniform(10.0, 40.0),
}

tuner = tune.Tuner(
    train_als_objective,
    tune_config=tune.TuneConfig(
        scheduler  = ASHAScheduler(metric="rmse", mode="min", grace_period=1),
        search_alg = OptunaSearch(metric="rmse", mode="min"),
        num_samples = 12,
    ),
)
```

**All 12 trial results (sorted by RMSE):**

| Run | Rank | Reg | Iter | RMSE | MAE | MAP@10 | Note |
|---|---|---|---|---|---|---|---|
| ray-tune-r150-reg0.05-i15 | 150 | 0.05 | 15 | 0.6404 | 0.4909 | 0.0474 | Best HPO trial |
| **als-rank100-reg0.01** (final) | **100** | **0.01** | **15** | **0.6410** | **0.4948** | **0.0466** | **Chosen model** |
| ray-tune-r200-reg0.01-i10 | 200 | 0.01 | 10 | 0.6458 | 0.4959 | 0.0453 | Marginal gain, needs more iters |
| ray-tune-r100-reg0.05-i10 | 100 | 0.05 | 10 | 0.6517 | 0.5034 | 0.0437 | Approaching optimal |
| ray-tune-r100-reg0.1-i10 | 100 | 0.1  | 10 | 0.6674 | 0.5130 | 0.0414 | Good rank, moderate reg |
| ray-tune-r50-reg0.01-i10  | 50  | 0.01 | 10 | 0.6820 | 0.5235 | 0.0345 | Mid rank, low reg |
| ray-tune-r50-reg0.1-i10   | 50  | 0.1  | 10 | 0.6902 | 0.5332 | 0.0307 | Mid rank improving |
| ray-tune-r200-reg0.3-i15  | 200 | 0.3  | 15 | 0.6766 | 0.5233 | 0.0333 | High rank wasted by reg |
| ray-tune-r150-reg0.3-i10  | 150 | 0.3  | 10 | 0.6866 | 0.5248 | 0.0318 | Rank=150 wasted by too much reg |
| ray-tune-r100-reg0.5-i10  | 100 | 0.5  | 10 | 0.7112 | 0.5462 | 0.0265 | Good rank, over-regularised |
| ray-tune-r50-reg0.5-i5    | 50  | 0.5  | 5  | 0.7461 | 0.5780 | 0.0189 | High reg kills signal |
| ray-tune-r10-reg0.01-i5   | 10  | 0.01 | 5  | 0.7545 | 0.5773 | 0.0168 | Too few latent dims |
| ray-tune-r10-reg0.1-i10   | 10  | 0.1  | 10 | 0.7628 | 0.5884 | 0.0155 | Low rank + high reg = underfit |

**Key findings:**

```
RMSE vs rank (best reg per rank):
  rank=10  → RMSE 0.754  ████████████████████████████
  rank=50  → RMSE 0.682  ██████████████████████░░░░░░
  rank=100 → RMSE 0.641  █████████████████████░░░░░░░
  rank=150 → RMSE 0.640  █████████████████████░░░░░░░  ← near-optimal
  rank=200 → RMSE 0.646  █████████████████████░░░░░░░

Diminishing returns beyond rank=100. Chosen model (rank=100, reg=0.01) is
computationally cheaper than rank=150/200 with equivalent quality.

Regularisation is critical: reg=0.5 degrades even rank=150 to RMSE=0.687.
Optimal range: reg ∈ [0.01, 0.05] for this dataset size.
```

The final model (`rank=100, reg=0.01, iter=15`) was chosen over the HPO best (`rank=150, reg=0.05`) because the RMSE difference (0.6410 vs 0.6404) is negligible while the rank=100 model trains 30% faster and uses ~30% less Redis memory.

### 5.5 Spark Structured Streaming

**File:** `ingestion/spark_streaming.py`

Three concurrent streaming queries consume the `user-events` Kafka topic:

**Query 1 — Trending feed (5-minute tumbling window):**

```python
events_with_watermark = events.withWatermark("event_time", "10 minutes")

trending_query = (
    events_with_watermark
    .filter(col("event_type").isin("click", "view", "rate"))
    .groupBy(window(col("event_time"), "5 minutes"), col("movie_id"))
    .agg(count("*").alias("event_count"))
    .writeStream
    .foreachBatch(write_trending_to_redis)   # top-20 → trending:realtime (TTL=5min)
    .trigger(processingTime="30 seconds")
    .start()
)
```

**Query 2 — Live rating aggregates (10-minute sliding window with 2-min slide):**

```python
ratings_query = (
    events_with_watermark
    .filter(col("event_type") == "rate")
    .groupBy(window(col("event_time"), "10 minutes", "2 minutes"), col("movie_id"))
    .agg(avg("rating").alias("stream_avg_rating"),
         count("*").alias("stream_rating_count"))
    .writeStream
    .foreachBatch(write_ratings_to_redis)    # → movie_feat:{id} hash (TTL=10min)
    .trigger(processingTime="60 seconds")
    .start()
)
```

**Query 3 — Bronze append (60-second micro-batches):**

```python
bronze_query = (
    events
    .writeStream
    .format("parquet")
    .option("path", "data/streaming_output/bronze/events")
    .option("checkpointLocation", "data/checkpoints/bronze")
    .partitionBy("event_type")
    .trigger(processingTime="60 seconds")
    .start()
)
```

**Demo without Kafka** (`ingestion/mock_streaming.py`): simulates the Spark Streaming output by writing directly to Redis every 30 seconds, using a popularity-weighted Poisson distribution of event counts. No Docker required.

### 5.6 Serving Layer — FastAPI + Redis

**File:** `serving/api.py`

The API loads two data structures into memory at startup and keeps them hot:

```python
# 32,722 rows × 10 columns — for search and metadata enrichment
_movies_df: pd.DataFrame = None

# 56,558 rows × 100 cols, L2-normalised — for similarity and from-likes
_item_factors: np.ndarray = None   # shape: (56558, 100), dtype=float32

# Pre-built search indices for rapidfuzz
_titles_lower: list[str]       # normalised titles
_title_words:  list[list[str]] # tokenised title words (stopwords removed)
```

**Search implementation (3-stage cascade):**

```
Stage 1: Exact / prefix / contains (pandas str ops, no fuzzy)
  - exact match on normalised title
  - starts-with match
  - substring contains match

Stage 2: Genre enrichment
  - normalised genre field contains query

Stage 3: Fuzzy fallback (rapidfuzz)
  - single-word query:  max(rfuzz.ratio(q, word) for word in title_words) >= 72
  - multi-word query:   rfuzz.token_sort_ratio(q, full_title) >= 65

Final scoring per result:
  score = 0.45 × fuzzy_match + 0.35 × (log(rating_count+1)/log(max_count+1)) + 0.20 × (avg_rating/5.0)
  (balances relevance with popularity so iconic titles rank first)
```

**Redis precompute** (`serving/redis_precompute.py`):

```python
# Step 1: Load ALS item factors (56,558 × 100 from parquet)
item_factors = load_item_factors()   # L2-normalise in place

# Step 2: Compute item-item similarities (sklearn NearestNeighbors, cosine)
nn = NearestNeighbors(n_neighbors=11, metric="cosine", algorithm="brute", n_jobs=-1)
nn.fit(item_factors)
distances, indices = nn.kneighbors(item_factors)  # ~2 min for 56K items

# Step 3: Load ALS user recommendations from Parquet
recs_df = spark.read.parquet("data/recommendations")  # model.recommendForAllUsers(20)

# Step 4: Push all keys to Redis (pipeline, 1000 keys/batch)
# → 162,541 recs:{user_id} + 56,558 similar:{movie_id} + popular_movies
# → Total: 219,099 keys, ~30 seconds
```

### 5.7 Frontend — Streamlit Netflix UI

**File:** `frontend/app.py`

The UI is a single-file Streamlit application with ~1,400 lines of Python + embedded CSS. Custom styling achieves a Netflix-like aesthetic:

```css
background-color: #0f0f0f;   /* near-black background */
color: #e8e8e8;               /* warm white text */
accent: #e50914;              /* Netflix red */
font-family: 'Inter', -apple-system, sans-serif;
card hover: scale(1.07) translateY(-4px)   /* subtle lift on hover */
```

**Tabs and features:**

| Tab | Content | Key Technical Feature |
|---|---|---|
| Home | Daily-rotating hero banner + 11 genre rows + personalised row | ALS user-vector solve on "from-likes"; hero cycles through top-8 posters by day-of-year |
| My List | Per-profile saved movies + "Because you added these" row | Independent list per profile (`all_liked_movies[user_id]`); real-time taste model update |
| Trending | Live event counts from Kafka stream | Redis `trending:realtime` key with 5-min TTL |
| Rate | Movie search with thumbnail grid + star rating + written review | POST /rate sends review text + username; search has "← Back" to clear results |
| Analytics | 9 Plotly charts + dataset KPIs + model metrics | Genre pie chart + Avg Rating by Year scatter shown first; reads Parquet + MLflow SQLite |

**Movie detail panel**: clicking the "Details" button on any card opens a full-width panel with:
- Large poster image
- Title, year, genres, average rating + rating count
- TMDB plot overview text
- **Community reviews** — 2-column grid of seeded and user-submitted reviews with star ratings, reviewer names, and dates (fetched from `GET /reviews/{movie_id}`)
- "Similar Movies" row (7 ALS item-item neighbours from Redis)
- "Add to My List" button (triggers real-time taste solve)

**Per-profile My List:** session state key `all_liked_movies` is a nested dict `{user_id: {movie_id: info}}`. Switching profiles (e.g. from Demo User A to Demo User C) instantly swaps to that profile's independent saved list. No cross-profile bleed.

**Community reviews system:**
- SQLite database at `data/reviews.db` — 147,336 pre-seeded reviews across 32,722 movies (3–6 per movie)
- Reviews generated deterministically from `data/seed_reviews.py` using `random.Random(movie_id * 31337)` so each movie always has the same reviewers
- Users can submit their own written review on the Rate tab; it is stored under their profile name
- Star ratings are weighted to match each movie's average rating (higher-rated movies get more 5-star reviews)

**Demo user presets** (for grading without a real user account):

| Profile | User ID | Taste profile |
|---|---|---|
| Demo User A | 42 | Action / Adventure fan |
| Demo User B | 500 | Drama / Art film fan |
| Demo User C | 7,777 | Comedy / Animation fan |
| Demo User D | 50,000 | Thriller / Crime fan |
| Demo User E | 120,000 | Sci-Fi fan |

Each profile maintains its own independent "My List". Adding movies to Demo User A does not affect Demo User B's list.

---

## 6. Live Demo

### Quick start (Redis + precomputed data already loaded)

```bash
# Terminal 1: FastAPI backend
REDIS_HOST=localhost REDIS_PORT=6379 DATA_DIR=$(pwd)/data \
python3 -m uvicorn serving.api:app --host 0.0.0.0 --port 8000

# Terminal 2: Streamlit frontend
API_URL=http://localhost:8000 \
streamlit run frontend/app.py --server.port 8501

# Terminal 3 (optional): Live streaming demo (no Kafka/Docker needed)
python3 ingestion/mock_streaming.py
```

**Verify everything is working:**

```
http://localhost:8501          Streamlit UI (Netflix-style)
http://localhost:8000/docs     FastAPI interactive docs
http://localhost:8000/health   → {"status":"ok","redis":true,"movies_loaded":32722}
http://localhost:5001          MLflow experiment tracker
```

### Demo walkthrough

**1. Home tab — personalised recommendations:**
- Select a Demo User from the dropdown (e.g. "Demo User A" = user 42)
- The "Top Picks for You" row loads precomputed ALS recs from Redis (< 5 ms)
- Click "Details" on any movie → see poster, overview, similar movies

**2. My List — real-time taste model:**
- Click "+ List" on several movies (different genres to see the interplay)
- The "Recommended For You" row appears and changes with each addition
- The taste model indicator shows: "1 title → weak" → "6+ titles → well-conditioned"
- This demonstrates real-time ALS user-vector solve without retraining

**3. Trending tab — live streaming data:**
- First run `python3 ingestion/mock_streaming.py` in a terminal
- The Trending tab shows movies with event counts and a "Live" badge
- Counts update every 30 seconds with new simulated window data

**4. Rate tab — submit a review:**
- Type a movie name in the search box → results appear as clickable thumbnail cards (not a dropdown)
- Click "Select" on a thumbnail to load that movie for rating
- Use the "← Back" button to clear the search and browse again
- Set a star rating (0.5–5.0), type an optional written review, and submit
- Your review immediately appears under "Details" for that movie

**5. Movie detail panel — community reviews:**
- Click "Details" on any movie card
- Scroll past the overview to see a 2-column grid of community reviews
- Each review shows reviewer name, star rating, date, and written text
- Reviews come from 147K pre-seeded entries plus any ratings users submit during the demo

**6. Analytics tab — data science story:**
- Dataset KPIs: 25M ratings, 162K users, 59K movies, 99.74% sparsity
- Model performance: RMSE=0.6410, MAP@10=0.0466, Precision@10=0.0521
- 9 live Plotly charts — first row shows Avg Rating by Year (survivorship bias) and a Genre Pie chart
- MLflow metrics pull from the SQLite database at runtime

**7. Search:**
- Type "shawshank" → "The Shawshank Redemption" appears first (fuzzy word-level match)
- Type "sci fi space" → Star Wars, Interstellar, Gravity ranked by relevance + popularity
- Type "comedy" → genre row with top-rated comedies
- Hit "← Back" to return to the full Home browse view

### Full pipeline (from scratch)

```bash
# Prerequisites: Python 3.11+, Spark 3.5, Docker Desktop, Redis

make setup        # pip install + mkdir data directories
make download     # wget MovieLens 25M (~650 MB)
make etl          # Spark ETL: ~8 min, outputs gold ratings + feature tables
# (optional) python3 data/tmdb_enrichment.py --limit 500
make train        # Spark MLlib ALS: ~12 min, outputs model + recs parquet
make precompute   # redis_precompute.py: ~30 sec, pushes 219K keys to Redis
make api-dev      # start FastAPI on port 8000
make frontend-dev # start Streamlit on port 8501
```

---

## 7. Results & Visualizations

### 7.1 Dataset Analytics

The Analytics tab (live in the app) renders these charts from the actual processed data. Figures below show the numbers these charts encode.

**Rating distribution (25,000,095 total):**

| Stars | Count | Share | Observation |
|---|---|---|---|
| 0.5 ★ | 393,068 | 1.6% | |
| 1.0 ★ | 776,815 | 3.1% | |
| 1.5 ★ | 399,490 | 1.6% | |
| 2.0 ★ | 1,640,868 | 6.6% | |
| 2.5 ★ | 1,262,797 | 5.1% | |
| 3.0 ★ | 4,896,928 | 19.6% | |
| 3.5 ★ | 3,177,318 | 12.7% | |
| **4.0 ★** | **6,639,798** | **26.6%** | **← most common** |
| 4.5 ★ | 2,200,539 | 8.8% | |
| 5.0 ★ | 3,612,474 | 14.4% | |

*Interpretation: The distribution peaks at 4.0 stars — users choose movies they expect to enjoy (selection bias). Mean rating = 3.53 stars. This left-skew is why implicit feedback mode (confidence weighting) outperforms explicit rating prediction.*

**Top 20 most-rated movies:**

| Rank | Title | Year | Ratings | Avg ★ | Genres |
|---|---|---|---|---|---|
| 1 | Forrest Gump | 1994 | 81,491 | 4.05 | Comedy, Drama, Romance |
| 2 | Shawshank Redemption, The | 1994 | 81,482 | 4.41 | Drama, Crime |
| 3 | Pulp Fiction | 1994 | 79,672 | 4.19 | Thriller, Crime, Comedy |
| 4 | Silence of the Lambs, The | 1991 | 74,127 | 4.15 | Crime, Thriller |
| 5 | Matrix, The | 1999 | 72,674 | 4.15 | Action, Science Fiction |
| 6 | Star Wars: Episode IV — A New Hope | 1977 | 68,717 | 4.12 | Adventure, Action, Sci-Fi |
| 7 | Jurassic Park | 1993 | 64,144 | 3.68 | Adventure, Science Fiction |
| 8 | Schindler's List | 1993 | 60,411 | 4.25 | Drama, History, War |
| 9 | Braveheart | 1995 | 59,184 | 4.00 | Action, Drama, History |
| 10 | Fight Club | 1999 | 58,773 | 4.23 | Drama, Thriller |

**Genre distribution (movies per genre):**

| Genre | Movies | Total Ratings | Avg ★ |
|---|---|---|---|
| Drama | 15,595 | ~9.8M | 3.35 |
| Comedy | 11,008 | ~6.2M | 3.25 |
| Thriller | 6,121 | ~5.1M | 3.30 |
| Romance | 5,821 | ~3.8M | 3.28 |
| Action | 4,720 | ~5.8M | 3.22 |
| Crime | 4,145 | ~4.9M | 3.42 |
| Horror | 4,003 | ~2.6M | 3.10 |
| Adventure | 3,156 | ~4.4M | 3.32 |
| Documentary | 2,767 | ~0.9M | 3.48 |
| Science Fiction | 2,508 | ~4.3M | 3.28 |

*Insight: Documentary, History, and War films earn the highest average ratings (~3.4–3.5), while Horror consistently lags (~3.1). This genre quality signal is captured by the ALS model — documentaries get lower confidence weights (fewer ratings) but higher implicit preference scores.*

**The Long-Tail Problem:**

```
Movie rank by popularity  →  Number of ratings received

Rank 1 (Forrest Gump):       81,491 ratings  ████████████████████████████████
Rank 2 (Shawshank):          81,482 ratings  ████████████████████████████████
Rank 10:                     58,773 ratings  ███████████████████████░░░░░░░░░
Rank 100:                    19,891 ratings  ████████░░░░░░░░░░░░░░░░░░░░░░░░
Rank 1,000:                   2,857 ratings  █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
Rank 5,000:                     438 ratings  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
Rank 10,000:                     81 ratings  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
Rank 32,722:                      5 ratings  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░

► Top 10% of movies (~3,272 titles) account for ~86% of all ratings.
► ALS collaborative filtering is essential to discover the 90% long-tail.
```

**User activity distribution:**

| Ratings given | Users | Share |
|---|---|---|
| 1–50 | 74,210 | 45.7% |
| 51–100 | 34,148 | 21.0% |
| 101–250 | 30,641 | 18.9% |
| 251–500 | 13,218 | 8.1% |
| 501–1,000 | 6,273 | 3.9% |
| 1,000–5,000 | 3,866 | 2.4% |
| 5,000+ | 185 | 0.1% |

*Insight: 66.7% of users rated fewer than 100 movies. The extreme sparsity of user-item interactions is exactly the problem ALS latent factor models are designed to solve.*

**Average rating by year (survivorship bias):**

Movies released before 1980 have systematically higher average ratings (~3.6–3.7) compared to recent films (~3.3–3.4). This is survivorship bias: only widely praised classics from that era remain in the catalogue and collect ongoing ratings. The LOWESS trendline in the Analytics tab makes this effect clearly visible.

### 7.2 Model Performance

**Final model (ALS rank=100, reg=0.01, α=25, iter=15, implicit=True):**

| Metric | Value | Interpretation |
|---|---|---|
| RMSE | **0.6410** | Error on binarised labels (rating ≥ 4.0 = positive preference) |
| MAE | **0.4948** | Mean absolute error on same basis |
| MAP@10 | **0.0466** | Mean average precision at K=10 (ranking quality) |
| Precision@10 | **0.0521** | Among top-10 recommended, 5.2% are rated ≥ 4.0 |
| Coverage | **0.847** | 84.7% of movies appear in at least one user's top-20 |
| Train size | 19,997,837 | 80% split |
| Test size | 5,002,258 | 20% split |
| Training time | ~12 minutes | Spark local[*], 10g driver, M1/M2 Mac |

**On MAP@10 = 0.0466:** This is consistent with published results on MovieLens 25M with implicit ALS. A value of 0.05 means that on average, 1 in 20 of the top-10 recommendations would be rated ≥ 4 stars by the user if they watched it. This seems low but reflects the fundamental difficulty of the task — we are recommending out of 59,047 items with only a few hundred ratings per user, and most "ground truth positives" in the test set were never rated (the zero-one missing-data problem in implicit CF).

### 7.3 Hyperparameter Sweep Results

See full table in [Section 5.4](#54-hyperparameter-optimisation-with-ray-tune).

**RMSE vs. key parameters (visual summary):**

```
Effect of rank (best reg per rank, all 15 iterations):
rank=10:  RMSE≈0.755  ████████████████████████████████
rank=50:  RMSE≈0.682  █████████████████████████████░░░
rank=100: RMSE≈0.641  ███████████████████████████░░░░░
rank=150: RMSE≈0.640  ███████████████████████████░░░░░  ← plateau
rank=200: RMSE≈0.646  ███████████████████████████░░░░░

Effect of regularisation (rank=100, 10 iterations):
reg=0.01:  RMSE≈0.651  ███████████████████████████░░░░░
reg=0.05:  RMSE≈0.652  ███████████████████████████░░░░░
reg=0.1:   RMSE≈0.667  ████████████████████████████░░░░
reg=0.3:   RMSE≈0.687  █████████████████████████████░░░
reg=0.5:   RMSE≈0.711  ██████████████████████████████░░
```

### 7.4 System Latency

| Endpoint | Mechanism | p50 | p99 |
|---|---|---|---|
| `GET /recommend/{user_id}` | Redis string get → JSON parse | < 3 ms | < 5 ms |
| `GET /similar/{movie_id}` | Redis string get → JSON parse | < 2 ms | < 3 ms |
| `GET /trending` | Redis string get → enrich with metadata | < 4 ms | < 8 ms |
| `GET /popular` | Redis string get | < 2 ms | < 4 ms |
| `POST /recommend/from-likes` (1 like) | 100×100 linear solve + 56K dot products | < 6 ms | < 10 ms |
| `POST /recommend/from-likes` (10 likes) | Same but better-conditioned system | < 8 ms | < 12 ms |
| `GET /movies/search` (fuzzy) | rapidfuzz on 32K titles | < 30 ms | < 50 ms |
| `GET /movies/{movie_id}` | pandas row lookup | < 5 ms | < 8 ms |
| `POST /rate` | validate + Kafka publish | < 15 ms | < 25 ms |

Redis precompute throughput: **219,099 keys in ~30 seconds** (~7,300 keys/sec).

### 7.5 Item Similarity Validation

Cosine similarity of ALS item factor vectors produces semantically meaningful results — the model has learned that stylistically similar movies cluster in the 100-dimensional latent space without any explicit content features:

| Seed Movie | Top 3 Similar (ALS cosine similarity) |
|---|---|
| Toy Story (1995) | Toy Story 2 (0.993), Toy Story 3 (0.983), A Bug's Life (0.976) |
| The Matrix (1999) | The Matrix Reloaded (0.991), Dark City (0.964), eXistenZ (0.958) |
| Forrest Gump (1994) | Rain Man (0.962), Field of Dreams (0.959), Good Will Hunting (0.944) |
| The Shawshank Redemption | The Green Mile (0.976), Schindler's List (0.967), The Godfather (0.959) |
| Star Wars: A New Hope | The Empire Strikes Back (0.994), Return of the Jedi (0.992), Raiders of the Lost Ark (0.978) |
| Pulp Fiction (1994) | Reservoir Dogs (0.988), Fargo (0.971), Trainspotting (0.960) |
| Schindler's List (1993) | The Pianist (0.972), Life Is Beautiful (0.963), Saving Private Ryan (0.961) |

The model clusters:
- Children's animated films (Pixar / DreamWorks)
- Mind-bending sci-fi (Matrix, Dark City, eXistenZ)
- Prestige drama (Shawshank, Green Mile, Schindler's List)
- Tarantino-style crime thrillers (Pulp Fiction, Reservoir Dogs, Fargo)
- The Star Wars saga (near-perfect cosine similarity between episodes)

This emergent semantic structure validates that the 100-dimensional ALS latent space has learned meaningful representations from interaction data alone — no genre tags or plot text were used in training.

---

## 8. Running the Application

### Prerequisites

| Tool | Version | Install |
|---|---|---|
| Python | 3.11+ | python.org |
| Apache Spark | 3.5.x | `/opt/spark` |
| Docker Desktop | 27+ | docker.com |
| Redis | 7+ | `brew install redis` |
| TMDB API key | — | themoviedb.org/settings/api (optional, for enrichment) |

### Step-by-step pipeline

```bash
# 1. Clone and install
git clone <repo-url>
cd MovieRecommendation
cp .env.example .env         # set TMDB_API_KEY if you want poster images
pip3 install -r requirements.txt

# 2. Download MovieLens 25M (~650 MB)
python3 data/download_datasets.py

# 3. Start infrastructure
brew services start redis    # Redis on port 6379
docker-compose up -d zookeeper kafka kafka-ui  # Kafka (optional, for streaming)

# 4. Spark ETL (~8 min)
make etl

# 5. (Optional) TMDB enrichment
python3 data/tmdb_enrichment.py --limit 500

# 6. Train ALS model (~12 min)
make train

# 7. Push 219K recs to Redis (~30 sec)
make precompute

# 8. Start services
make api-dev          # FastAPI on :8000
make frontend-dev     # Streamlit on :8501
mlflow server \       # MLflow UI on :5001
  --host 0.0.0.0 --port 5001 \
  --backend-store-uri "sqlite:///$(pwd)/mlflow/mlflow.db" \
  --default-artifact-root "$(pwd)/mlflow/artifacts" &

# 9. (Optional) Live streaming demo
python3 ingestion/mock_streaming.py   # no Kafka needed
# OR with full Kafka:
make stream-producer   # Terminal A
make stream-consumer   # Terminal B
```

### Service URLs

| Service | URL | Purpose |
|---|---|---|
| Streamlit UI | http://localhost:8501 | Main application |
| FastAPI docs | http://localhost:8000/docs | Interactive API explorer |
| MLflow UI | http://localhost:5001 | Experiment tracking, model registry |
| Kafka UI | http://localhost:8090 | Topic and message browser |
| Redis CLI | `redis-cli -p 6379 keys '*' | wc -l` | Verify 219K keys |

### Makefile shortcuts

```bash
make setup          # install deps + create directories
make download       # download datasets
make etl            # Spark ETL
make train          # ALS training
make precompute     # push recs to Redis
make api-dev        # FastAPI server
make frontend-dev   # Streamlit UI
make tune           # Ray Tune HPO sweep
make stream-demo    # mock streaming (no Kafka)
make stream-producer # Kafka event producer
make stream-consumer # Spark streaming consumer
make clean          # remove __pycache__, checkpoints
```

---

## 9. API Reference

### `GET /recommend/{user_id}?n=20`

Returns top-N personalised recommendations from Redis cache.

```json
[{
  "movie_id": 356, "title": "Forrest Gump", "score": 4.048,
  "genres": "Comedy|Drama|Romance|War",
  "poster_url": "https://image.tmdb.org/t/p/w342/...",
  "avg_rating": 4.05, "rating_count": 81491
}]
```

Fallback: returns popular movies for unknown users (cold start).

---

### `POST /recommend/from-likes`

Solves ALS update equation for the user's taste vector and returns top-N picks.

**Body:** `{"likes": [{"movie_id": 1, "weight": 1.0}, ...], "n": 10}`

Latency: < 10 ms. Each additional like refines the taste vector.

---

### `GET /similar/{movie_id}?n=10`

Returns movies similar by ALS item-factor cosine similarity (from Redis).

---

### `GET /movies/search?q=shawshank&limit=20`

Three-stage fuzzy search: exact → prefix → rapidfuzz word-level ratio.

---

### `GET /movies/{movie_id}`

Full metadata: title, year, genres, poster_url, overview, avg_rating, rating_count. Also returns live stream features (`stream_avg_rating`) if available.

---

### `GET /trending?n=20`

Reads `trending:realtime` from Redis (Spark Streaming output). Falls back to popular movies if no stream is running.

---

### `GET /popular?n=20`

Top movies by rating count, cached in Redis.

---

### `POST /rate`

**Body:**
```json
{
  "user_id": 1,
  "movie_id": 296,
  "rating": 4.5,
  "review": "A timeless classic that holds up beautifully.",
  "username": "Demo User A"
}
```

`review` and `username` are optional. Validates input, publishes to Kafka `user-events`, saves review to `data/reviews.db` (upsert on `(user_id, movie_id)`), and invalidates the user's Redis cache.

---

### `GET /reviews/{movie_id}?limit=50`

Returns community reviews for a movie, newest first.

```json
[{
  "user_id": 42,
  "rating": 4.5,
  "review": "A timeless classic that holds up beautifully.",
  "username": "Demo User A",
  "created_at": "2026-05-01T19:30:00"
}]
```

The database is pre-seeded with 147,336 reviews across 32,722 movies (3–6 per movie) via `data/seed_reviews.py`. Reviews use realistic human names and rating-weighted text templates.

---

### `GET /health`

```json
{"status": "ok", "redis": true, "kafka": false, "movies_loaded": 32722}
```

---

## 10. Project Structure

```
MovieRecommendation/
│
├── data/
│   ├── movielens/ml-25m/          ← 25M ratings + metadata (download_datasets.py)
│   ├── enriched/tmdb_metadata.parquet   ← TMDB poster URLs + overviews
│   ├── processed/
│   │   ├── gold/ratings/          ← ALS-ready parquet (25M rows)
│   │   ├── movie_features/        ← title, genres, year, poster, avg_rating (32K rows)
│   │   └── user_features/         ← rating_count, avg_rating per user (162K rows)
│   ├── recommendations/           ← model.recommendForAllUsers(20) parquet
│   ├── similar_movies/            ← model.recommendForAllItems(10) parquet
│   ├── reviews.db                 ← SQLite: 147K seeded reviews + user submissions
│   ├── download_datasets.py
│   ├── tmdb_enrichment.py
│   └── seed_reviews.py            ← seeds 3-6 human-named reviews per movie (deterministic)
│
├── ingestion/
│   ├── kafka_producer.py          ← simulated user events → Kafka (configurable rate)
│   ├── spark_streaming.py         ← Kafka consumer: trending windows → Redis
│   ├── mock_streaming.py          ← demo streaming without Kafka
│   └── init_topics.py             ← create Kafka topics on first run
│
├── processing/
│   ├── etl_batch.py               ← Spark ETL: bronze→silver→gold (AQE enabled)
│   └── feature_engineering.py    ← genre one-hot, rating bias, implicit confidence
│
├── ml/
│   ├── train_als.py               ← Spark MLlib ALS + MLflow logging + precompute
│   └── ray_tune_als.py            ← Ray Tune HPO: ASHAScheduler + OptunaSearch
│
├── serving/
│   ├── api.py                     ← FastAPI: 8 endpoints, Redis + ALS solver
│   ├── redis_precompute.py        ← bulk push 219K keys to Redis after training
│   ├── Dockerfile
│   └── requirements.txt
│
├── frontend/
│   ├── app.py                     ← Streamlit Netflix UI (~1400 lines)
│   ├── Dockerfile
│   └── requirements.txt
│
├── mlflow/
│   ├── mlflow.db                  ← SQLite: 14 runs (1 initial, 1 final, 12 HPO)
│   └── artifacts/
│       └── als-model-local/       ← Spark model: itemFactors (56K×100), userFactors
│
├── docker-compose.yml             ← Kafka, Zookeeper, Kafka UI, Postgres, Redis, API, UI
├── Makefile                       ← pipeline shortcuts
├── requirements.txt               ← Python dependencies
├── .env.example                   ← TMDB_API_KEY, REDIS_HOST, KAFKA_BOOTSTRAP_SERVERS
└── README.md
```

---

## 11. Conclusion

### Summary

CineRec demonstrates a complete, production-grade movie recommendation system built on a modern big-data stack. Starting from 25 million raw interaction records, the system processes data through a Spark ETL pipeline, trains a collaborative filtering model that learns 100-dimensional representations for every user and movie, and serves personalised recommendations at < 10 ms latency through a Redis-backed FastAPI layer. A Spark Structured Streaming pipeline consumes live user events from Kafka and updates a trending feed in real time with 5-minute tumbling windows. The entire ML lifecycle is tracked in MLflow with 14 experiment runs including a 12-trial Ray Tune hyperparameter sweep.

**What was built end-to-end:**

| Component | Technology | Scale |
|---|---|---|
| Batch ETL | Spark (AQE) | 25M rows, ~8 min |
| Collaborative filtering | Spark MLlib ALS | 162K users × 59K movies, ~12 min |
| Hyperparameter search | Ray Tune + Optuna | 12 trials, ASHAScheduler |
| Experiment tracking | MLflow | 14 runs, model registry |
| Real-time streaming | Kafka + Spark Streaming | 50 events/sec, 5-min windows |
| Serving | FastAPI + Redis | 219K precomputed keys, < 5 ms |
| Online personalisation | ALS update equation | Real-time user vector solve, < 10 ms |
| Search | rapidfuzz | 3-stage cascade on 32K titles |
| UI | Streamlit | Netflix-style, 5 tabs, live demo |

### Lessons Learned

**1. Implicit feedback is harder to evaluate than it looks.**  
RMSE = 0.64 sounds good but means little when labels are binarised (0/1). MAP@10 = 0.047 is the honest number — and it reflects a genuinely hard task: recommending 10 movies out of 59,047 that a user would rate ≥ 4 stars, given only ~71 ratings as prior.

**2. Redis is the right serving layer for precomputed recs.**  
The alternative — running model inference at request time — would require keeping Spark alive (500 MB JVM overhead) just for serving. Redis decouples training from serving completely. The `from-likes` online solver is the right escape hatch for real-time personalisation.

**3. AQE transforms Spark on uneven data.**  
The rating matrix is heavily skewed: a handful of popular movies have 80K+ ratings while 75% of movies have fewer than 200. AQE's skew-join optimization and dynamic partition coalescing reduced ETL runtime from ~20 minutes (fixed 200 partitions) to ~8 minutes.

**4. Search needs a cascade, not a single scorer.**  
The original difflib-based search ranked "W." above "Shawshank Redemption" for query "shawshnk" because `partial_ratio` finds the string "w" inside the query as a perfect match. Moving to word-level `rfuzz.ratio` comparisons (each query word vs each title word, take the max) fixed this class of false positive entirely.

**5. Sparsity is the core challenge.**  
The 99.74% sparse rating matrix is what makes simple approaches fail. A nearest-neighbour system would have no neighbours for 75% of users. ALS factorises the sparse matrix into dense embeddings that generalise from observed to unobserved interactions — this is the whole point.

### Future Work

| Enhancement | Complexity | Impact |
|---|---|---|
| **Two-Tower neural model** (user + item encoders) | High | Better accuracy on sparse users; handles content features naturally |
| **Sequential recommendation** (SASRec / BERT4Rec) | High | Models watch-order and recent session context |
| **Multi-armed bandit for exploration** | Medium | Balances recommending known-good vs. discovering new titles for each user |
| **Real-time model updates** (online ALS) | Medium | Incorporate new ratings without full retraining; currently 24-hour staleness |
| **A/B testing framework** | Medium | Compare model versions on live traffic; currently all users see the same model |
| **Cold-start content features** | Medium | Use genre, director, cast embeddings (from TMDB) for new movies with no ratings |
| **Distributed Redis cluster** | Low | Horizontal scale for production traffic; current setup is single-node |
| **Kubernetes deployment** | Low | Replace Docker Compose with K8s manifests for production ops |
| **Diversity and fairness constraints** | High | Prevent filter bubbles; ensure long-tail movies get surfaced proportionally |

---

*Built for SJSU Big Data · May 2026*  
*Dataset: MovieLens 25M — GroupLens Research, University of Minnesota*  
*Model: Spark MLlib ALS (collaborative filtering, implicit feedback mode)*
