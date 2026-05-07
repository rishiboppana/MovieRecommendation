.PHONY: setup infra-up infra-down download etl train precompute stream api frontend all clean

SPARK_SUBMIT=/opt/spark/bin/spark-submit
SPARK_PACKAGES=org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0
PYTHON=python3
SPARK_COMMON=--conf spark.driver.bindAddress=127.0.0.1 --conf spark.driver.host=127.0.0.1
export PYSPARK_PYTHON=python3
export PYSPARK_DRIVER_PYTHON=python3

# ---------- Setup ----------
setup:
	cp -n .env.example .env || true
	$(PYTHON) -m pip install -r requirements.txt
	mkdir -p data/amazon data/movielens data/enriched mlflow/artifacts

# ---------- Infrastructure ----------
mlflow-server:
	mlflow server \
		--host 0.0.0.0 --port 5001 \
		--backend-store-uri "sqlite:///$(PWD)/mlflow/mlflow.db" \
		--default-artifact-root "$(PWD)/mlflow/artifacts" \
		--serve-artifacts &
	@echo "MLflow UI: http://localhost:5001"

infra-up:
	docker-compose up -d zookeeper kafka kafka-ui postgres redis
	@echo "Waiting for services..."
	@sleep 15
	$(PYTHON) ingestion/init_topics.py
	$(MAKE) mlflow-server

infra-down:
	docker-compose down

infra-clean:
	docker-compose down -v

# ---------- Data ----------
download:
	$(PYTHON) data/download_datasets.py

enrich:
	$(PYTHON) data/tmdb_enrichment.py

# ---------- Processing ----------
etl:
	$(SPARK_SUBMIT) \
		--master local[*] \
		--driver-memory 8g \
		$(SPARK_COMMON) \
		--conf spark.sql.adaptive.enabled=true \
		--conf spark.sql.adaptive.coalescePartitions.enabled=true \
		--conf spark.sql.adaptive.skewJoin.enabled=true \
		processing/etl_batch.py

feature-eng:
	$(SPARK_SUBMIT) \
		--master local[*] \
		--driver-memory 8g \
		$(SPARK_COMMON) \
		processing/feature_engineering.py

# ---------- ML ----------
train:
	$(SPARK_SUBMIT) \
		--master local[*] \
		--driver-memory 10g \
		$(SPARK_COMMON) \
		--conf spark.sql.adaptive.enabled=true \
		ml/train_als.py

tune:
	$(PYTHON) ml/ray_tune_als.py

# ---------- Serving ----------
precompute:
	$(PYTHON) serving/redis_precompute.py

api-dev:
	cd serving && uvicorn api:app --host 0.0.0.0 --port 8000 --reload

# ---------- Streaming ----------
stream-producer:
	$(PYTHON) ingestion/kafka_producer.py

stream-consumer:
	$(SPARK_SUBMIT) \
		--master local[2] \
		--packages $(SPARK_PACKAGES) \
		--conf spark.sql.streaming.checkpointLocation=./data/checkpoints \
		ingestion/spark_streaming.py

stream-demo:
	@echo "Starting mock streaming (no Kafka/Docker required)..."
	$(PYTHON) ingestion/mock_streaming.py

# ---------- Frontend ----------
frontend-dev:
	cd frontend && streamlit run app.py --server.port 8501

# ---------- Docker all-in-one ----------
up:
	docker-compose up -d

down:
	docker-compose down

# ---------- Full pipeline ----------
pipeline: infra-up download etl train precompute
	@echo "Pipeline complete. Run 'make up' to start all services."

# ---------- Clean ----------
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf data/checkpoints
