# CineRec — Large-Scale Movie Recommendation System

**SJSU CS Big Data · Semester 3 Project**  
**Student:** Rishi Visweswar Boppana · rishivisweswar.boppana@sjsu.edu

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Datasets](#3-datasets)
4. [Technology Stack](#4-technology-stack)
5. [Data Pipeline](#5-data-pipeline)
6. [Machine Learning Model](#6-machine-learning-model)
7. [Real-Time Streaming](#7-real-time-streaming)
8. [Serving Layer](#8-serving-layer)
9. [Frontend Application](#9-frontend-application)
10. [MLflow Experiment Tracking](#10-mlflow-experiment-tracking)
11. [Running the Application](#11-running-the-application)
12. [API Reference](#12-api-reference)
13. [Project Structure](#13-project-structure)
14. [Results & Metrics](#14-results--metrics)
15. [Key Technical Decisions](#15-key-technical-decisions)

---

## 1. Project Overview

CineRec is a production-grade movie recommendation system built on a modern big-data stack. It ingests over **32 million ratings** from two public datasets, trains a collaborative-filtering model using **Spark MLlib ALS** on the full dataset, and serves personalised recommendations through a **FastAPI** backend cached in **Redis**. A real-time user-event stream flows through **Apache Kafka** into **Spark Structured Streaming**, keeping a live trending feed updated. Every training run is logged and versioned in **MLflow**.

The system demonstrates the complete machine learning lifecycle at scale:

```
Raw data  →  Spark ETL  →  ALS Training  →  MLflow Registry
                                  ↓
               Redis cache ←  Precomputed recs
                    ↓
Kafka stream  →  Spark Streaming  →  Redis trending
                    ↓
           FastAPI  →  Streamlit UI (Netflix-style)
```

### What makes this non-trivial

| Dimension | Detail |
|---|---|
| Data volume | 25 M MovieLens ratings + 7.4 M Amazon ratings = **32.4 M total** |
| Users | 162,541 unique users |
| Items | 59,047 unique movies |
| Model size | 100-dim latent vectors × (162 K users + 59 K movies) ≈ 22 M parameters |
| Cache | 219,100 Redis keys — every user has precomputed top-20 recs |
| Inference | ALS user-vector solve in < 5 ms at request time (no model reload) |

---

## 2. System Architecture

```
┌─────────────────────────── Data Sources ───────────────────────────┐
│  MovieLens 25M          Amazon Reviews 2023      TMDB API          │
│  ratings + metadata     Movies & TV (7.4M)       posters + genres  │
└──────────┬──────────────────────┬────────────────────┬─────────────┘
           │  Batch               │  Batch             │  Enrichment
           ▼                      ▼                    ▼
┌─────────────────────────── Ingestion Layer ────────────────────────┐
│                                                                     │
│   Kafka  ◄── kafka_producer.py (simulated user events)             │
│   Topic: user-events                                                │
│                                                                     │
│   Data Lake (Parquet)  ←  raw CSVs / JSONLs                        │
└──────────┬──────────────────────┬─────────────────────────────────┘
           │                      │
           ▼                      ▼
┌─────────────────────────── Processing Layer ───────────────────────┐
│                                                                     │
│   Spark ETL (AQE enabled)          Spark Structured Streaming       │
│   ├── Bronze: raw parquet          ├── Consume user-events topic    │
│   ├── Silver: cleaned + joined     ├── 5-min tumbling windows       │
│   └── Gold: ALS-ready matrix       └── Write trending → Redis       │
│                                                                     │
│   Spark Catalog (in-memory)                                         │
│   └── Registered views: gold_ratings, movie_features, user_features│
└──────────┬─────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────── ML Layer ──────────────────────────────┐
│                                                                     │
│   Spark MLlib ALS  (implicit feedback mode)                         │
│   ├── rank=100, α=25, λ=0.01, iterations=15                        │
│   ├── 25M ratings as confidence weights: c = 1 + 25 × rating       │
│   └── Learns 100-dim latent vectors for every user + movie          │
│                                                                     │
│   Ray Tune  (distributed hyperparameter search)                     │
│   └── ASHAScheduler, OptunaSearch, 20 trials                        │
│                                                                     │
│   MLflow Tracking                                                    │
│   ├── Experiment: movie-recommendations-als                         │
│   ├── Logs: params, metrics, model artifact                         │
│   └── Model Registry: movie-rec-als v1 → Production                │
└──────────┬─────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────── Serving Layer ─────────────────────────┐
│                                                                     │
│   Redis (219 K keys)                                                │
│   ├── recs:{user_id}    → top-20 precomputed picks (162 K keys)    │
│   ├── similar:{movie_id}→ top-10 similar movies  (56 K keys)       │
│   ├── popular_movies    → top-1000 by rating count                 │
│   └── trending:realtime → live Kafka-aggregated trending           │
│                                                                     │
│   FastAPI (8 endpoints)                                             │
│   ├── Loads ALS item-factor matrix (56 K × 100) into memory        │
│   ├── Serves recs from Redis in < 5 ms                             │
│   └── ALS user-vector solver for "from-likes" in real time         │
└──────────┬─────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────── Frontend ──────────────────────────────┐
│   Streamlit — Netflix-style UI                                      │
│   ├── Home: hero banner + genre rows + personalised picks           │
│   ├── My List: liked movies → ALS taste solver → recommendations   │
│   ├── Trending: live Kafka stream results                           │
│   └── Rate: submit rating → Kafka → real-time update               │
└────────────────────────────────────────────────────────────────────┘
```

---

## 3. Datasets

### 3.1 MovieLens 25M

| File | Size | Content |
|---|---|---|
| `ratings.csv` | 647 MB | 25,000,095 ratings by 162,541 users on 59,047 movies |
| `movies.csv` | 2.9 MB | Movie titles and pipe-separated genres |
| `links.csv` | 1.3 MB | MovieLens → IMDb → TMDB ID mapping |
| `tags.csv` | 37 MB | Free-text user tags per movie |
| `genome-scores.csv` | 415 MB | Tag relevance scores for 13,000+ tags across all movies |

Source: [grouplens.org/datasets/movielens/25m](https://grouplens.org/datasets/movielens/25m/)

### 3.2 Amazon Reviews 2023 — Movies & TV

| File | Size | Content |
|---|---|---|
| `Movies_and_TV_ratings.csv` | 412 MB | 7,441,129 ratings (user_id, asin, rating, timestamp) |
| `Movies_and_TV_reviews.jsonl` | ~2.6 GB | Full review text, images, verified-purchase flag |

Source: [amazon-reviews-2023.github.io](https://amazon-reviews-2023.github.io) via HuggingFace  
`McAuley-Lab/Amazon-Reviews-2023`

### 3.3 TMDB API Enrichment

- 500+ movies enriched with poster URLs, overview text, popularity scores, runtime
- API key from [themoviedb.org/settings/api](https://www.themoviedb.org/settings/api)
- Stored as `data/enriched/tmdb_metadata.parquet`

### 3.4 Combined dataset size

| Layer | Size on disk |
|---|---|
| Raw inputs | ~4.8 GB |
| Spark-processed parquet (silver + gold) | ~500 MB |
| Model artifacts (item/user factors) | ~171 MB |
| **Total** | **~5.5 GB** |

The full Amazon Reviews JSONL (18 GB uncompressed) satisfies the 10 GB requirement when retained. It was removed during development to free shuffle space for ALS training.

---

## 4. Technology Stack

| Layer | Technology | Version | Purpose |
|---|---|---|---|
| Data processing | Apache Spark | 3.5.8 | ETL, ALS training, streaming |
| ML | Spark MLlib ALS | 3.5.8 | Collaborative filtering |
| HPO | Ray Tune + Optuna | 2.20.0 | Hyperparameter search |
| Experiment tracking | MLflow | 3.11.1 | Run logging, model registry |
| Message broker | Apache Kafka | 7.5.0 (Confluent) | Real-time event streaming |
| Stream processing | Spark Structured Streaming | 3.5.8 | Kafka consumer + aggregation |
| Cache | Redis | 7 / 8.6.2 | Precomputed rec cache |
| API | FastAPI + Uvicorn | 0.111.0 | REST serving layer |
| Frontend | Streamlit | 1.33.0 | Netflix-style UI |
| Orchestration | Docker Compose | 27.5.1 | Service orchestration |
| Language | Python | 3.13 | All application code |
| Compute | macOS (local) | — | Development + demo |

---

## 5. Data Pipeline

### 5.1 Batch ETL (`processing/etl_batch.py`)

The ETL runs on Spark with **Adaptive Query Execution (AQE)** enabled, which automatically:
- Coalesces shuffle partitions based on actual data statistics
- Detects and handles skewed joins
- Optimises physical plans at runtime

**Configuration:**
```python
spark.sql.adaptive.enabled                       = true
spark.sql.adaptive.coalescePartitions.enabled    = true
spark.sql.adaptive.skewJoin.enabled              = true
spark.sql.shuffle.partitions                     = 100
```

**Stages:**

```
Bronze  →  Raw parquet from CSV/JSONL sources
           (MovieLens ratings, Amazon ratings, TMDB metadata)

Silver  →  Cleaned and joined:
           - MovieLens ratings joined with movie metadata
           - TMDB poster URLs and genres merged in
           - Filtered: rating ∈ [0.5, 5.0], non-null user/movie IDs
           - 25,000,095 rows

Gold    →  ALS-ready matrix:
           - Deduplicated: one row per (user_id, movie_id) pair
           - Integer IDs: user_id ∈ [1, 162541], movie_id ∈ [1, 209171]
           - Saved as Parquet, registered as Spark Catalog view "gold_ratings"
```

**Additional processed tables:**

| View | Content |
|---|---|
| `movie_features` | movie_id, title, genres, avg_rating, rating_count, poster_url |
| `user_features` | user_id, rating_count, avg_rating, min/max rating |

### 5.2 Feature Engineering (`processing/feature_engineering.py`)

- **Genre one-hot encoding**: 19 genre columns (Action, Comedy, Drama, …)
- **Rating bias decomposition**: global mean, user bias, item bias, residual rating
- **Implicit confidence**: `c_ui = 1 + α × r_ui` (used in implicit ALS)

---

## 6. Machine Learning Model

### 6.1 Algorithm: Alternating Least Squares (ALS)

ALS is a matrix factorisation algorithm for collaborative filtering. It decomposes the sparse user-item rating matrix **R** (162 K × 59 K, 99.74% sparse) into two dense matrices:

```
R ≈ U × Vᵀ

U ∈ ℝ^(n_users × rank)   — user latent factor matrix
V ∈ ℝ^(n_items × rank)   — item latent factor matrix
```

Each row of **U** is a 100-dimensional "taste vector" for a user.  
Each row of **V** is a 100-dimensional "identity vector" for a movie.

The predicted rating for user `u` on movie `i` is:

```
r̂_ui = uᵀ · vᵢ
```

**Objective function (implicit feedback mode):**

```
min_{U,V}  Σ_{u,i} c_ui (p_ui − uᵀvᵢ)² + λ(||U||² + ||V||²)

where:
  p_ui = 1  (user interacted with movie i)
  c_ui = 1 + α × r_ui   (confidence weight; α=25)
  λ    = regularisation parameter (0.01)
```

The alternating step fixes **V** and solves for each row of **U** in closed form (least squares), then fixes **U** and solves for **V**. Each solve is a small linear system solvable in O(rank²) time, making the whole algorithm parallelisable across Spark partitions.

### 6.2 Why Implicit Feedback Mode

Explicit ALS predicts the exact numerical rating (1–5 stars).  
Implicit ALS treats ratings as **confidence signals**: a higher rating means the user is *more likely* to want this movie, not that they would rate it 5 stars. This is more appropriate for recommendation because:

1. Most users never rate movies they watch — implicit data is far more abundant
2. A rating of 3 does not mean the user dislikes the film — it could just be average
3. Confidence weighting (`c = 1 + 25 × rating`) naturally down-weights uncertain signals

### 6.3 Hyperparameters

| Parameter | Value | Meaning |
|---|---|---|
| `rank` | 100 | Dimensionality of latent factor vectors |
| `regParam` | 0.01 | L2 regularisation strength |
| `maxIter` | 15 | Number of ALS alternating steps |
| `alpha` | 25.0 | Confidence scaling factor for implicit feedback |
| `implicitPrefs` | True | Use implicit feedback objective |
| `coldStartStrategy` | drop | Exclude unknown users/items from evaluation |

### 6.4 Training at Scale

```
Dataset  : 25,000,095 ratings  (80% train / 20% test split)
Train    : ~20,000,000 ratings
Test     : ~5,000,000 ratings
Runtime  : ~12 minutes on local Spark (local[*], 8-core, 10 GB driver)
```

**Spark submit command:**
```bash
spark-submit \
  --master local[*] \
  --driver-memory 10g \
  --conf spark.driver.bindAddress=127.0.0.1 \
  --conf spark.sql.adaptive.enabled=true \
  --conf spark.sql.shuffle.partitions=100 \
  --conf spark.local.dir=/tmp/spark-scratch \
  ml/train_als.py \
  --rank 100 --reg-param 0.01 --max-iter 15 --alpha 25.0
```

### 6.5 Evaluation Metrics

| Metric | Value |
|---|---|
| RMSE (test set) | 0.8092 |
| MAE (test set) | 0.6306 |

Note: RMSE/MAE are computed on binarised labels (≥4.0 = positive) in implicit mode, which is the standard evaluation approach for implicit collaborative filtering.

### 6.6 Real-Time User Vector Solve ("Add to My List" feature)

When a user adds movies to their list, we do not retrain the model. Instead we solve for the optimal user latent vector given their liked movies using the ALS update equation:

```
u* = (XᵀCX + λI)⁻¹ · Xᵀc

where:
  X   = item factor matrix for liked movies  (n_likes × 100)
  C   = diagonal confidence matrix           diag(1 + α × weight_i)
  c   = confidence vector                    (1 + α × weight_i) for each like
  λ   = 0.1 regularisation

Time: < 5 ms (numpy linear solve on a (100×100) system)
```

This gives a proper taste-vector recommendation — not just nearest-neighbour lookup. Each new liked movie re-solves the system, refining the taste vector:

- 1 like → vector points at that movie's embedding
- 2 likes → vector balances both; overlap region floats up
- 5+ likes → well-conditioned system; confident, genre-aware recommendations

**Scoring all movies against the taste vector:**
```python
scores = item_factor_matrix @ user_vector   # (56558,) — one dot product per movie
```

This is a single matrix-vector multiply: O(n_movies × rank) = O(5.6 M) operations, completing in < 3 ms.

### 6.7 Item-to-Item Similarity

Similarity between movies is computed as cosine similarity between item factor vectors. The item factor matrix `V` (56,558 × 100) is L2-normalised at startup, so:

```
similarity(i, j) = vᵢᵀ · vⱼ   (dot product of normalised vectors = cosine sim)
```

`sklearn.neighbors.NearestNeighbors` (brute-force, cosine metric) finds top-10 similar movies for all 56,558 items. Results are stored in Redis under `similar:{movie_id}`.

---

## 7. Real-Time Streaming

### 7.1 Kafka Event Flow

```
kafka_producer.py  →  Kafka topic: user-events  →  spark_streaming.py
      ↑                                                     ↓
  POST /rate                                            Redis keys:
  (FastAPI endpoint)                                    trending:realtime
                                                        movie_feat:{id}
```

**Event schema:**
```json
{
  "user_id": 12345,
  "movie_id": 296,
  "event_type": "rate",
  "rating": 4.5,
  "session_id": 839201,
  "timestamp": "2026-05-01T19:30:00Z"
}
```

**Topics:**
| Topic | Producer | Consumer | Purpose |
|---|---|---|---|
| `user-events` | kafka_producer.py, POST /rate | spark_streaming.py | All user events |
| `rating-updates` | POST /rate | (future) | Rating-specific events |
| `trending-updates` | (reserved) | (future) | Aggregated trending signals |

### 7.2 Spark Structured Streaming (`ingestion/spark_streaming.py`)

Three concurrent streaming queries run in parallel:

**Query 1 — Trending movies (5-min tumbling window)**
```python
events
  .filter(event_type IN ('click', 'view', 'rate'))
  .groupBy(window(event_time, "5 minutes"), movie_id)
  .agg(count("*").alias("event_count"))
  .writeStream
  .foreachBatch(write_trending_to_redis)   # top-20 → Redis TTL 5 min
  .trigger(processingTime="30 seconds")
```

**Query 2 — Live rating aggregates (10-min sliding window)**
```python
events
  .filter(event_type == 'rate')
  .groupBy(window(event_time, "10 minutes", "2 minutes"), movie_id)
  .agg(avg("rating"), count("*"))
  .writeStream
  .foreachBatch(write_ratings_to_redis)    # movie_feat:{id} → Redis
  .trigger(processingTime="60 seconds")
```

**Query 3 — Bronze layer (append to parquet)**
```python
events
  .writeStream
  .format("parquet")
  .partitionBy("event_type")              # data/streaming_output/bronze/events
  .trigger(processingTime="60 seconds")
```

### 7.3 Starting the streaming pipeline

```bash
# Terminal 1: produce 50 events/sec for continuous demo
python3 ingestion/kafka_producer.py --rate 50

# Terminal 2: Spark Structured Streaming consumer
spark-submit \
  --master local[2] \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 \
  ingestion/spark_streaming.py
```

The Trending tab in the UI will update with real event counts within ~30 seconds.

---

## 8. Serving Layer

### 8.1 FastAPI (`serving/api.py`)

**Startup sequence:**
1. Connect to Redis — ping to verify
2. `load_movies_df()` — read `data/processed/movie_features` parquet → pandas DataFrame (32,720 movies)
3. `load_item_factors()` — read ALS item-factor parquet → numpy array (56,558 × 100), L2-normalise in place
4. `init_kafka()` — connect to Kafka producer for `POST /rate`

**Request routing:**

```
GET /recommend/{user_id}   Redis lookup → filter ghost IDs → return top-N
GET /similar/{movie_id}    Redis lookup → enrich with metadata
POST /recommend/from-likes  ALS user-vector solve → numpy score all items
GET /movies/search          pandas string search on title + genres
GET /movies/{movie_id}      metadata lookup + Redis stream features
GET /trending               Redis trending:realtime or popular fallback
GET /popular                Redis popular_movies (top-1000 by rating count)
GET /health                 service status
POST /rate                  validate → Kafka publish → invalidate user cache
```

### 8.2 Redis Precompute (`serving/redis_precompute.py`)

After training, all recommendations are precomputed and pushed to Redis:

```
162,541  user recommendation sets     → recs:{user_id}          (TTL: 24 h)
 56,558  item similarity sets         → similar:{movie_id}       (TTL: 24 h)
  1,000  popular movies (cached)      → popular_movies            (TTL: 24 h)
─────────────────────────────────────────────────────────────────
219,099  total Redis keys
```

Item similarities use sklearn `NearestNeighbors` (brute-force cosine) on the item factor matrix — no Spark needed, completes in ~2 minutes.

---

## 9. Frontend Application

### 9.1 Overview

Netflix-style dark UI built in Streamlit with custom CSS. No emojis; Inter font, `#0f0f0f` background, `#e50914` red accent colour.

### 9.2 Tabs

**Home**
- Full-width hero banner with movie poster background
- "Recommended For You" row (ALS taste model, only shown when list has items)
- "Top Picks for You" row (precomputed per user_id from Redis)
- 11 genre rows: Drama, Action, Comedy, Science Fiction, Thriller, Romance, Horror, Crime, Animation, Documentary, War
- "Most Popular" row

**My List**
- All movies added via "Add to My List" button
- "Because You Added These" row — real-time ALS taste solve
- Taste model quality indicator (1 title = weak → 6+ titles = well-conditioned)

**Trending**
- Live data from Kafka stream aggregated by Spark
- Falls back to popular movies when no stream is running

**Rate**
- Submit explicit rating for any movie ID
- Published to Kafka; explanation of the full pipeline shown inline

### 9.3 "Add to My List" — How It Works

1. Click **Add to My List** on any movie card
2. Movie stored in Streamlit session state with its ALS score as a confidence weight
3. On next render, `POST /recommend/from-likes` is called with all liked movies
4. API solves the ALS update equation for the user's taste vector (< 5 ms)
5. Scores all 56,558 movies against that vector (< 3 ms)
6. Filters to movies with real metadata only
7. Returns top-10 as the "Recommended For You" row

Each additional liked movie refines the taste vector — recommendations visibly improve from the second or third like onwards.

---

## 10. MLflow Experiment Tracking

**UI:** `http://localhost:5001`

**Experiment:** `movie-recommendations-als`

| Field | Value |
|---|---|
| Tracking URI | `sqlite:///./mlflow/mlflow.db` |
| Artifact root | `./mlflow/artifacts/` |
| Registered model | `movie-rec-als` |
| Model version | v1 |
| Stage | Production |
| Alias | champion |

**Logged per run:**

| Type | Fields |
|---|---|
| Parameters | rank, reg_param, max_iter, alpha, mode (implicit/explicit), spark_version, dataset |
| Metrics | rmse, mae, train_size, test_size |
| Artifacts | `als-model/` — MLmodel YAML, conda.yaml, sparkml/ (item + user factors) |
| Tags | model_registered, model_stage, model_alias |

**Start the MLflow server:**
```bash
mlflow server \
  --host 0.0.0.0 --port 5001 \
  --backend-store-uri "sqlite:///$(pwd)/mlflow/mlflow.db" \
  --default-artifact-root "$(pwd)/mlflow/artifacts"
```

---

## 11. Running the Application

### 11.1 Prerequisites

| Tool | Version | Install |
|---|---|---|
| Python | 3.13 | python.org |
| Apache Spark | 3.5.8 | `/opt/spark` (pre-installed) |
| Docker Desktop | 27.5+ | docker.com |
| Homebrew Redis | 8.6+ | `brew install redis` |
| TMDB API key | — | themoviedb.org/settings/api |

### 11.2 First-time Setup

```bash
git clone <repo>
cd project

# Install Python dependencies
pip3 install -r requirements.txt

# Copy and edit config
cp .env.example .env
# Set TMDB_API_KEY in .env

# Create required directories
mkdir -p data/amazon data/movielens data/enriched mlflow/artifacts
```

### 11.3 Step-by-step Pipeline

**Step 1 — Download datasets**
```bash
# MovieLens 25M + Amazon ratings (~700 MB total, ~10 min)
python3 data/download_datasets.py --no-full-reviews

# For full 10 GB dataset (Amazon Reviews with text, ~30 min download):
python3 data/download_datasets.py
```

**Step 2 — Start infrastructure**
```bash
# Option A: Docker (Kafka + Kafka UI + Postgres)
docker-compose up -d zookeeper kafka kafka-ui postgres

# Option B: Redis locally (if Docker Desktop is unavailable)
/usr/local/opt/redis/bin/redis-server --daemonize yes --port 6379
```

**Step 3 — Create Kafka topics**
```bash
python3 ingestion/init_topics.py
```

**Step 4 — Run Spark ETL**
```bash
PYSPARK_PYTHON=python3 PYSPARK_DRIVER_PYTHON=python3 \
spark-submit \
  --master local[*] \
  --driver-memory 8g \
  --conf spark.driver.bindAddress=127.0.0.1 \
  --conf spark.sql.adaptive.enabled=true \
  --conf spark.sql.shuffle.partitions=100 \
  processing/etl_batch.py
# Output: data/processed/{gold/ratings, movie_features, user_features}
```

**Step 5 — (Optional) TMDB enrichment**
```bash
python3 data/tmdb_enrichment.py --limit 500   # fast: 500 movies
python3 data/tmdb_enrichment.py               # full: all 62K movies (~1 hour)
```

**Step 6 — Train ALS model**
```bash
PYSPARK_PYTHON=python3 PYSPARK_DRIVER_PYTHON=python3 \
spark-submit \
  --master local[*] \
  --driver-memory 10g \
  --conf spark.driver.bindAddress=127.0.0.1 \
  --conf spark.sql.adaptive.enabled=true \
  --conf spark.local.dir=/tmp/spark-scratch \
  ml/train_als.py \
  --rank 100 --reg-param 0.01 --max-iter 15 --alpha 25.0
# Runtime: ~12 minutes
# Output: mlflow/artifacts/als-model-local/, data/recommendations/
```

**Step 7 — Push recommendations to Redis**
```bash
python3 serving/redis_precompute.py
# Pushes 219K keys in ~30 seconds
```

**Step 8 — Start MLflow UI**
```bash
mlflow server \
  --host 0.0.0.0 --port 5001 \
  --backend-store-uri "sqlite:///$(pwd)/mlflow/mlflow.db" \
  --default-artifact-root "$(pwd)/mlflow/artifacts" &
```

**Step 9 — Start API**
```bash
REDIS_HOST=localhost REDIS_PORT=6379 \
DATA_DIR=$(pwd)/data \
python3 -m uvicorn serving.api:app --host 0.0.0.0 --port 8000
```

**Step 10 — Start Streamlit UI**
```bash
API_URL=http://localhost:8000 \
python3 -m streamlit run frontend/app.py \
  --server.port 8501 --server.address 0.0.0.0 \
  --server.headless true
```

**Step 11 — (Optional) Start streaming demo**
```bash
# Terminal A: produce events
python3 ingestion/kafka_producer.py --rate 50

# Terminal B: consume + aggregate
spark-submit \
  --master local[2] \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 \
  ingestion/spark_streaming.py
```

### 11.4 Service URLs

| Service | URL | Purpose |
|---|---|---|
| Streamlit UI | http://localhost:8501 | Main application |
| FastAPI | http://localhost:8000/docs | Interactive API docs |
| MLflow | http://localhost:5001 | Experiment tracking |
| Kafka UI | http://localhost:8090 | Topic browser |
| Redis CLI | `redis-cli -p 6379` | Cache inspection |

### 11.5 Makefile shortcuts

```bash
make setup          # install deps + create directories
make infra-up       # start Docker services + Kafka topics
make download       # download all datasets
make etl            # run Spark ETL
make train          # train ALS model
make precompute     # push recs to Redis
make stream-producer # start event producer (Kafka)
make stream-consumer # start Spark streaming consumer
make tune           # Ray Tune hyperparameter search
make up             # docker-compose up all services
make clean          # remove __pycache__, checkpoints
```

---

## 12. API Reference

### `GET /recommend/{user_id}`
Returns top-N personalised movie recommendations from Redis cache.

**Parameters:** `n` (int, default 10, max 50)

**Logic:**
1. Look up `recs:{user_id}` in Redis
2. Filter to movies with real metadata (removes ghost IDs from sparse regions)
3. Enrich each result with title, genres, poster URL
4. If no cached recs (new user): return popular movies

**Response:**
```json
[
  {
    "movie_id": 356,
    "title": "Forrest Gump",
    "score": 4.048,
    "genres": "Comedy|Drama|Romance|War",
    "poster_url": "https://image.tmdb.org/t/p/w342/...",
    "avg_rating": 4.05,
    "rating_count": 81491
  }
]
```

---

### `POST /recommend/from-likes`
Solves for the user's optimal ALS latent vector and returns top-N personalised picks.

**Body:**
```json
{
  "likes": [
    {"movie_id": 1,   "weight": 1.0},
    {"movie_id": 356, "weight": 1.0}
  ],
  "n": 10
}
```

**Logic (ALS update equation):**
```
u* = (XᵀCX + λI)⁻¹ · Xᵀc
scores = item_factor_matrix · u*   (56,558-dim dot products)
```

Filters ghost IDs, returns top-N with real metadata only.

---

### `GET /similar/{movie_id}`
Returns movies similar to the given movie based on cosine similarity of ALS item factors.

---

### `POST /rate`
Publishes a user rating to Kafka topic `user-events` and invalidates the user's Redis cache.

**Body:** `{"user_id": 1, "movie_id": 296, "rating": 4.5}`

---

### `GET /movies/search?q=...`
Full-text search across titles and genres using pandas string matching.

---

### `GET /trending`
Returns trending movies from `trending:realtime` Redis key (populated by Spark Streaming).
Falls back to popular movies if no stream is running.

---

### `GET /health`
```json
{
  "status": "ok",
  "redis": true,
  "kafka": false,
  "movies_loaded": 32720
}
```

---

## 13. Project Structure

```
project/
├── data/
│   ├── amazon/
│   │   └── Movies_and_TV_ratings.csv    (7.4M Amazon ratings)
│   ├── movielens/
│   │   └── ml-25m/                       (25M ratings + metadata)
│   ├── enriched/
│   │   └── tmdb_metadata.parquet         (TMDB poster URLs + genres)
│   ├── processed/
│   │   ├── gold/ratings/                 (ALS-ready parquet)
│   │   ├── movie_features/               (title, genres, poster_url, avg_rating)
│   │   └── user_features/                (rating_count, avg_rating)
│   ├── recommendations/                  (precomputed top-20 per user)
│   ├── download_datasets.py
│   └── tmdb_enrichment.py
│
├── ingestion/
│   ├── kafka_producer.py                 (simulated user events)
│   ├── spark_streaming.py                (Kafka consumer + Redis writer)
│   └── init_topics.py                    (create Kafka topics)
│
├── processing/
│   ├── etl_batch.py                      (Spark ETL: bronze→silver→gold)
│   └── feature_engineering.py           (genre encoding, bias decomposition)
│
├── ml/
│   ├── train_als.py                      (Spark MLlib ALS + MLflow)
│   └── ray_tune_als.py                   (distributed HPO with Ray Tune)
│
├── serving/
│   ├── api.py                            (FastAPI: 8 endpoints)
│   ├── redis_precompute.py               (bulk Redis population)
│   ├── Dockerfile
│   └── requirements.txt
│
├── frontend/
│   ├── app.py                            (Streamlit Netflix UI)
│   ├── Dockerfile
│   └── requirements.txt
│
├── mlflow/
│   ├── mlflow.db                         (SQLite tracking store)
│   └── artifacts/
│       └── als-model-local/              (Spark model: itemFactors, userFactors)
│
├── docker-compose.yml                    (Kafka, Postgres, Redis, API, UI)
├── Makefile
├── requirements.txt
├── .env.example
└── README.md
```

---

## 14. Results & Metrics

### Model performance

| Metric | Value | Notes |
|---|---|---|
| RMSE | 0.8092 | On binarised labels (≥4.0 = positive), implicit mode |
| MAE | 0.6306 | Same basis |
| Training data | 20,000,076 ratings | 80% split |
| Test data | 5,000,019 ratings | 20% split |
| Training time | ~12 minutes | Spark local[*], 10 GB driver, 15 iterations |

### System throughput

| Operation | Latency |
|---|---|
| `GET /recommend/{user_id}` (Redis hit) | < 5 ms |
| `POST /recommend/from-likes` (ALS solve) | < 10 ms |
| `GET /similar/{movie_id}` (Redis hit) | < 3 ms |
| `GET /movies/search` (pandas) | < 50 ms |
| Redis precompute (219K keys) | ~30 seconds |
| Spark ETL (25M ratings) | ~8 minutes |

### Item similarity validation (qualitative)

| Seed Movie | Top Similar |
|---|---|
| Toy Story | Toy Story 2 (0.993), Toy Story 3 (0.983), A Bug's Life (0.976), Finding Nemo (0.973) |
| The Matrix | The Matrix Reloaded, Dark City, eXistenZ, Strange Days |
| Forrest Gump | Rain Man (0.962), Field of Dreams (0.959), Good Will Hunting |

The similarity results confirm the ALS item factors have learned meaningful semantic structure — animated children's films cluster together, as do mind-bending sci-fi films.

---

## 15. Key Technical Decisions

### Why ALS and not a neural model?

ALS was chosen because:
1. **Scale**: Spark MLlib ALS is the standard choice for billion-scale collaborative filtering (Netflix Prize winner was ALS-based)
2. **Interpretability**: latent factor vectors can be inspected and arithmetically combined (taste vector solve)
3. **Speed**: training 25M ratings in 12 minutes on a single laptop — a neural approach would require a GPU cluster
4. **Integration**: native Spark model slots directly into the data pipeline with no additional serving infrastructure

A neural approach (e.g., Neural Collaborative Filtering, Two-Tower model) would give better accuracy at the cost of complexity and compute.

### Why implicit feedback mode?

Explicit ALS optimises for predicting star ratings — not for "will the user watch this?" Implicit ALS treats any rating as a positive preference signal, scaled by confidence. This better captures real recommendation behaviour and avoids the cold-start problem for users who rate infrequently.

### Why Redis over serving from the model directly?

ALS precomputes recommendations for all 162K users. Serving from Redis:
- **Latency**: < 5 ms vs 50–500 ms for model inference
- **Isolation**: API does not need Spark or a model server running
- **Simplicity**: Redis is operationally simple and horizontally scalable

The trade-off is staleness: recommendations are only as fresh as the last precompute run. For real-time personalisation, the `from-likes` endpoint solves a user vector on-demand using the pre-loaded item factor matrix.

### Why Kafka for a demo with simulated events?

Kafka makes the architecture production-ready:
- Decouples event producers from consumers — a real mobile app, web app, and recommendation engine can all share the same topic
- Provides durability (events survive consumer crashes)
- Enables exactly-once delivery semantics for rating updates
- The Spark Structured Streaming consumer demonstrates a production-grade pipeline pattern

### Why Streamlit over React?

Streamlit allows the entire frontend to be written in Python, keeping the tech stack consistent and minimising context-switching during development. The custom CSS brings it close enough to a Netflix-style look for demo purposes. A production system would use a React or Next.js frontend communicating with the same FastAPI backend.

---

*Built for SJSU Big Data Sem 3 · May 2026*
