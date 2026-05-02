"""
Spark Structured Streaming — consume 'user-events' topic from Kafka.

Responsibilities:
  - Parse and validate incoming events
  - Aggregate: rolling click counts, rating streams per movie
  - Write real-time feature updates to Redis (user activity, trending movies)
  - Write aggregated micro-batch summaries to parquet (bronze → streaming layer)

Submit:
    spark-submit \
        --master local[2] \
        --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 \
        ingestion/spark_streaming.py
"""
import os
import json
import redis
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, from_json, window, count, avg, max as spark_max,
    expr, current_timestamp, to_timestamp, lit
)
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType, TimestampType
)
from dotenv import load_dotenv

load_dotenv()

KAFKA_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
CHECKPOINT_DIR = "./data/checkpoints"
OUTPUT_DIR = "./data/streaming_output"

EVENT_SCHEMA = StructType([
    StructField("user_id", IntegerType(), True),
    StructField("movie_id", IntegerType(), True),
    StructField("event_type", StringType(), True),
    StructField("rating", DoubleType(), True),
    StructField("session_id", IntegerType(), True),
    StructField("timestamp", StringType(), True),
])


def write_trending_to_redis(df, epoch_id):
    """Write top trending movies in the current micro-batch to Redis."""
    rows = df.collect()
    if not rows:
        return
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    trending = [{"movie_id": row["movie_id"], "event_count": row["event_count"]} for row in rows[:20]]
    r.setex("trending:realtime", 300, json.dumps(trending))  # 5-min TTL
    r.close()


def write_ratings_to_redis(df, epoch_id):
    """Update running average ratings in Redis."""
    rows = df.collect()
    if not rows:
        return
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    pipe = r.pipeline()
    for row in rows:
        pipe.hset(f"movie_feat:{row['movie_id']}", mapping={
            "stream_avg_rating": round(row["avg_rating"], 3),
            "stream_rating_count": row["rating_count"],
        })
    pipe.execute()
    r.close()


def main():
    spark = (
        SparkSession.builder
        .appName("MovieRec-Streaming")
        .config("spark.sql.streaming.checkpointLocation", CHECKPOINT_DIR)
        .config("spark.streaming.stopGracefullyOnShutdown", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    raw = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_SERVERS)
        .option("subscribe", "user-events")
        .option("startingOffsets", "latest")
        .option("failOnDataLoss", "false")
        .load()
    )

    events = (
        raw.select(
            from_json(col("value").cast("string"), EVENT_SCHEMA).alias("data")
        )
        .select("data.*")
        .withColumn("event_time", to_timestamp(col("timestamp")))
        .withWatermark("event_time", "10 minutes")
    )

    # ── Stream 1: trending movies (5-min tumbling window) ──
    trending = (
        events
        .filter(col("event_type").isin("click", "view", "rate"))
        .groupBy(window(col("event_time"), "5 minutes"), col("movie_id"))
        .agg(count("*").alias("event_count"))
        .select("movie_id", "event_count")
        .orderBy(col("event_count").desc())
    )

    trending_query = (
        trending.writeStream
        .outputMode("complete")
        .foreachBatch(write_trending_to_redis)
        .trigger(processingTime="30 seconds")
        .option("checkpointLocation", f"{CHECKPOINT_DIR}/trending")
        .start()
    )

    # ── Stream 2: live rating aggregates ──
    rating_agg = (
        events
        .filter((col("event_type") == "rate") & col("rating").isNotNull())
        .groupBy(window(col("event_time"), "10 minutes", "2 minutes"), col("movie_id"))
        .agg(
            avg("rating").alias("avg_rating"),
            count("*").alias("rating_count"),
        )
        .select("movie_id", "avg_rating", "rating_count")
    )

    rating_query = (
        rating_agg.writeStream
        .outputMode("update")
        .foreachBatch(write_ratings_to_redis)
        .trigger(processingTime="60 seconds")
        .option("checkpointLocation", f"{CHECKPOINT_DIR}/ratings")
        .start()
    )

    # ── Stream 3: write raw events to parquet (bronze layer) ──
    bronze_query = (
        events.writeStream
        .outputMode("append")
        .format("parquet")
        .option("path", f"{OUTPUT_DIR}/bronze/events")
        .option("checkpointLocation", f"{CHECKPOINT_DIR}/bronze")
        .trigger(processingTime="60 seconds")
        .partitionBy("event_type")
        .start()
    )

    print("Streaming queries started. Ctrl+C to stop.")
    spark.streams.awaitAnyTermination()


if __name__ == "__main__":
    main()
