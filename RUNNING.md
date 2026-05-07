## Running CineRec (MovieRecommendation)

This document explains how to run the project (Docker and local), which environment variables are used, whether a `.env` is needed, and common setup steps (Kafka topics, Redis precompute, ETL & training). Keep this file next to the project root for quick reference.

## Quick summary

- For the easiest route run everything with Docker Compose (recommended for a full stack demo).
- A `.env` file at the project root is optional but recommended for local overrides and for providing secrets (e.g. `TMDB_API_KEY`). Docker Compose will read environment variables from the shell or a `.env` file when substituting ` ${VAR}` values in `docker-compose.yml`.
- Several Python scripts also call `load_dotenv()` so a `.env` in the repo root is used when running services locally.

## Required / useful environment variables

Put these in a `.env` at the project root for local runs or to provide values when using `docker-compose`.

- TMDB_API_KEY  (required for TMDB enrichment / poster URLs in the API)
- KAFKA_BOOTSTRAP_SERVERS  (default used in code: `localhost:29092`)
- REDIS_HOST  (default: `localhost`)
- REDIS_PORT  (default: `6379`)
- DATA_DIR  (default: `./data`) — location of processed parquet and recommendations
- MLFLOW_TRACKING_URI  (optional; docker-compose sets `http://mlflow:5001` inside the compose network)
- API_URL  (useful for local frontend; defaults to `http://localhost:8000` in code)

Example `.env` (create at project root):

```bash
TMDB_API_KEY=your_tmdb_api_key_here
KAFKA_BOOTSTRAP_SERVERS=localhost:29092
REDIS_HOST=localhost
REDIS_PORT=6379
DATA_DIR=./data
MLFLOW_TRACKING_URI=http://localhost:5001
API_URL=http://localhost:8000
```

Notes:
- `TMDB_API_KEY` is the only secret-like value required for TMDB enrichment and the API container (the `api` service in `docker-compose.yml` uses `${TMDB_API_KEY}`).
- If you plan to run Kafka/Redis via Docker Compose, keep the defaults above. When running services inside the compose network, services talk to each other using the service names (e.g. `redis`, `kafka`, `mlflow`).

## Option A — Quick start with Docker Compose (recommended)

1. Put a `.env` in the project root with at least `TMDB_API_KEY`.
2. Build & start the stack:

```bash
docker-compose up --build
```

3. What this starts (most important):
- Zookeeper & Kafka
- Redis
- Postgres + MLflow server
- API (FastAPI) on port `8000`
- Frontend (Streamlit) on port `8501`

4. After the stack is up:
- Create Kafka topics (one-time) — run `ingestion/init_topics.py` from your host Python environment with `KAFKA_BOOTSTRAP_SERVERS=localhost:29092` (or run inside a container that has network access to Kafka). Example:

```bash
# from repo root (host) after docker-compose up
export KAFKA_BOOTSTRAP_SERVERS=localhost:29092
python ingestion/init_topics.py
```

- If you have trained artifacts (ALS item factors and user recommendations) and want the API to serve precomputed recs and similarity data, run the Redis precompute script (it expects Redis accessible at `REDIS_HOST:REDIS_PORT` and data/artifacts mounted into the container or available on the host):

```bash
export REDIS_HOST=localhost
export REDIS_PORT=6379
python serving/redis_precompute.py --ttl 86400 --batch-size 5000
```

5. Health & UI:
- API health: http://localhost:8000/health
- Streamlit UI: http://localhost:8501
- Kafka UI (kafka-ui): http://localhost:8090

## Option B — Local development (no Docker)

1. Create a Python virtual environment and install dependencies (project has several requirements files; the root `requirements.txt` is a good start):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Create a `.env` file in the project root with the variables from the example above.

3. Start Redis and Kafka (you can use Docker for just these services or run locally). If you expose Kafka to the host, the project defaults expect `localhost:29092` for `KAFKA_BOOTSTRAP_SERVERS`.

4. Start the API locally:

```bash
# from project root
uvicorn serving.api:app --host 0.0.0.0 --port 8000 --reload
```

5. Start the frontend locally (Streamlit):

```bash
# from project root
streamlit run frontend/app.py --server.port 8501
```

6. Create Kafka topics and run `serving/redis_precompute.py` as described in the Docker section when you have artifacts.

## ETL / Training / Precompute order (high level)

1. Run batch ETL to produce the processed feature store (movie features, gold ratings): `processing/etl_batch.py` (Spark). This creates `data/processed/...` expected by the API.
2. Train ALS: `ml/train_als.py` (Spark). This writes model artifacts (itemFactors) into the MLflow artifact store path used by `serving/redis_precompute.py`.
3. Run `serving/redis_precompute.py` to push precomputed user recs and item similarities to Redis.
4. Start the API and frontend (or use Docker Compose) to serve the app.

Notes:
- Many of the above training/etl steps use Spark and are not one-line Python scripts — consult the `README.md` training sections for Spark submit commands and cluster resource recommendations.

## Where the code reads env vars

- `serving/api.py` uses `dotenv.load_dotenv()` and environment variables: `REDIS_HOST`, `REDIS_PORT`, `KAFKA_BOOTSTRAP_SERVERS`, and `DATA_DIR`.
- `ingestion/init_topics.py` and `ingestion/kafka_producer.py` read `KAFKA_BOOTSTRAP_SERVERS`.
- `serving/redis_precompute.py` reads `REDIS_HOST`/`REDIS_PORT`.
- `frontend/app.py` reads `API_URL` (default `http://localhost:8000`).

Because the code uses `python-dotenv` (`load_dotenv()`), a `.env` in the project root will be picked up automatically for local runs. Docker Compose substitutes `${VAR}` from the host environment or from a `.env` file too.

## Common troubleshooting

- Kafka connect errors: ensure Kafka is up (`docker-compose logs kafka`) and use `KAFKA_BOOTSTRAP_SERVERS=localhost:29092` when calling host-side scripts.
- Redis connection refused: check `REDIS_HOST`/`REDIS_PORT` and that Redis is healthy in compose (`docker-compose ps` / `docker-compose logs redis`).
- Missing movie features or item factors: the API will warn and fall back to cold-start/popular results. Run the ETL and training steps to populate `data/processed/movie_features` and `mlflow/artifacts/als-model-local/itemFactors`.

## Minimal checklist (quick)

- [ ] Create `.env` with `TMDB_API_KEY` (and other optional overrides).
- [ ] Start stack: `docker-compose up --build` (or follow local dev steps).
- [ ] Create Kafka topics: `python ingestion/init_topics.py` (host or container with network access).
- [ ] (Optional) Run training & `serving/redis_precompute.py` to pre-warm Redis with recs/similarity.

## Where to read more

See the main `README.md` for architecture, datasets, and Spark training details.

---

If you'd like, I can also add a minimal `.env.example` to the repo, or create a short Makefile target that runs the topic creation and precompute steps for you.
