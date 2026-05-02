"""
Spark Batch ETL — Bronze → Silver → Gold layers.

Reads:
  - MovieLens 25M: ratings.csv, movies.csv, tags.csv, links.csv
  - Amazon Reviews: Movies_and_TV_ratings.jsonl.gz (or full reviews)
  - TMDB metadata: enriched/tmdb_metadata.parquet

Produces:
  - Silver: cleaned, joined ratings + movie metadata
  - Gold: ALS-ready user-item matrix registered in Spark Catalog

Submit:
    spark-submit \
        --master local[*] \
        --driver-memory 8g \
        --conf spark.sql.adaptive.enabled=true \
        processing/etl_batch.py
"""
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, lit, trim, lower, split, explode,
    count, avg, stddev, min as spark_min, max as spark_max,
    year, to_date, regexp_extract, monotonically_increasing_id,
    concat_ws, collect_list
)
from pyspark.sql.types import IntegerType, FloatType
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
MOVIELENS_DIR = DATA_DIR / "movielens" / "ml-25m"
AMAZON_DIR = DATA_DIR / "amazon"
ENRICHED_DIR = DATA_DIR / "enriched"
OUTPUT_DIR = DATA_DIR / "processed"

WAREHOUSE_DIR = str(ROOT / "data" / "spark-warehouse")


def create_spark():
    return (
        SparkSession.builder
        .appName("MovieRec-ETL")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes", "256m")
        .config("spark.sql.warehouse.dir", WAREHOUSE_DIR)
        .config("spark.sql.catalogImplementation", "in-memory")
        .config("spark.driver.memory", "8g")
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .getOrCreate()
    )


# ─────────────────────────── BRONZE ───────────────────────────
def load_movielens(spark):
    print("Loading MovieLens 25M...")
    ratings = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(str(MOVIELENS_DIR / "ratings.csv"))
        .withColumnRenamed("userId", "ml_user_id")
        .withColumnRenamed("movieId", "ml_movie_id")
        .withColumnRenamed("rating", "ml_rating")
        .withColumnRenamed("timestamp", "ml_timestamp")
        .repartition(100)
    )
    print(f"  ML Ratings: {ratings.count():,} rows")

    movies = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(str(MOVIELENS_DIR / "movies.csv"))
        .withColumnRenamed("movieId", "ml_movie_id")
        .withColumn("year", regexp_extract(col("title"), r"\((\d{4})\)", 1).cast(IntegerType()))
        .withColumn("title_clean", trim(regexp_extract(col("title"), r"^(.*?)\s*\(\d{4}\)", 1)))
    )
    print(f"  ML Movies: {movies.count():,} rows")

    links = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(str(MOVIELENS_DIR / "links.csv"))
        .withColumnRenamed("movieId", "ml_movie_id")
    )

    tags = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(str(MOVIELENS_DIR / "tags.csv"))
        .withColumnRenamed("movieId", "ml_movie_id")
        .withColumnRenamed("userId", "ml_user_id")
        .withColumnRenamed("tag", "tag")
        .select("ml_movie_id", "tag")
        .filter(col("tag").isNotNull())
    )
    movie_tags = (
        tags.groupBy("ml_movie_id")
        .agg(concat_ws("|", collect_list("tag")).alias("tags"))
    )

    return ratings, movies, links, movie_tags


def load_amazon_ratings(spark):
    # Try CSV first (downloaded from HuggingFace), then jsonl.gz fallback
    csv_path = str(AMAZON_DIR / "Movies_and_TV_ratings.csv")
    jsonl_path = str(AMAZON_DIR / "Movies_and_TV_ratings.jsonl.gz")
    reviews_path = str(AMAZON_DIR / "Movies_and_TV_reviews.jsonl")

    if Path(csv_path).exists():
        print("Loading Amazon Reviews ratings (CSV)...")
        df = (
            spark.read
            .option("header", "true")
            .option("inferSchema", "true")
            .csv(csv_path)
            .select(
                col("user_id").alias("amz_user_id"),
                col("parent_asin").alias("asin"),
                col("rating").cast(FloatType()).alias("amz_rating"),
                col("timestamp").alias("amz_timestamp"),
            )
            .filter(col("amz_rating").isNotNull())
            .filter(col("amz_rating").between(1, 5))
        )
        print(f"  Amazon Ratings: {df.count():,} rows")
        return df
    elif Path(reviews_path).exists():
        print("Loading Amazon Reviews (full JSONL)...")
        df = (
            spark.read.json(reviews_path)
            .select(
                col("user_id").alias("amz_user_id"),
                col("parent_asin").alias("asin"),
                col("rating").cast(FloatType()).alias("amz_rating"),
                col("timestamp").alias("amz_timestamp"),
            )
            .filter(col("amz_rating").isNotNull())
            .filter(col("amz_rating").between(1, 5))
        )
        print(f"  Amazon Reviews: {df.count():,} rows")
        return df
    elif Path(jsonl_path).exists() and Path(jsonl_path).stat().st_size > 1000:
        print("Loading Amazon Reviews ratings (jsonl.gz)...")
        df = (
            spark.read.json(jsonl_path)
            .select(
                col("user_id").alias("amz_user_id"),
                col("parent_asin").alias("asin"),
                col("rating").cast(FloatType()).alias("amz_rating"),
                col("timestamp").alias("amz_timestamp"),
            )
            .filter(col("amz_rating").isNotNull())
            .filter(col("amz_rating").between(1, 5))
        )
        print(f"  Amazon Ratings: {df.count():,} rows")
        return df
    else:
        print("  Amazon ratings not found, skipping.")
        return None


def load_tmdb_metadata(spark):
    tmdb_path = str(ENRICHED_DIR / "tmdb_metadata.parquet")
    if not Path(tmdb_path).exists():
        print("  TMDB metadata not found, skipping enrichment.")
        return None
    print("Loading TMDB metadata...")
    df = spark.read.parquet(tmdb_path)
    print(f"  TMDB: {df.count():,} movies")
    return df


# ─────────────────────────── SILVER ───────────────────────────
def build_silver_ratings(ratings, movies, links, movie_tags, tmdb_df):
    """Join ratings with movie metadata → silver layer."""
    movie_meta = movies.join(links, "ml_movie_id", "left").join(movie_tags, "ml_movie_id", "left")
    if tmdb_df is not None:
        movie_meta = movie_meta.join(
            tmdb_df.select("movie_id", "poster_url", "overview", "popularity", "genres_str"),
            movie_meta["ml_movie_id"] == tmdb_df["movie_id"],
            "left"
        ).drop("movie_id")

    joined = (
        ratings
        .join(movie_meta, "ml_movie_id", "inner")
        .filter(col("ml_rating").isNotNull())
        .filter(col("ml_rating").between(0.5, 5.0))
    )
    # genres_str only present when TMDB metadata was loaded
    if "genres_str" in joined.columns:
        silver = joined.withColumn(
            "genres_final", when(col("genres_str").isNotNull(), col("genres_str")).otherwise(col("genres"))
        )
    else:
        silver = joined.withColumn("genres_final", col("genres"))
    return silver


# ─────────────────────────── GOLD ─────────────────────────────
def build_gold_als_matrix(silver_df):
    """Create ALS-ready user-item rating matrix."""
    # MovieLens IDs are already integers — perfect for ALS
    gold = (
        silver_df
        .select(
            col("ml_user_id").alias("user_id"),
            col("ml_movie_id").alias("movie_id"),
            col("ml_rating").alias("rating"),
            col("ml_timestamp").alias("timestamp"),
        )
        .filter(col("user_id").isNotNull() & col("movie_id").isNotNull())
        .dropDuplicates(["user_id", "movie_id"])
    )
    return gold


def build_movie_features(silver_df):
    """Aggregate movie-level features for cold-start and ranking."""
    # Only include TMDB columns if they were joined in
    available = set(silver_df.columns)
    group_cols = ["ml_movie_id", "title_clean", "genres_final", "year"]
    for optional in ["poster_url", "popularity", "overview"]:
        if optional in available:
            group_cols.append(optional)
        else:
            silver_df = silver_df.withColumn(optional, lit(None).cast("string"))
            group_cols.append(optional)

    return (
        silver_df
        .groupBy(*group_cols)
        .agg(
            count("*").alias("rating_count"),
            avg("ml_rating").alias("avg_rating"),
            stddev("ml_rating").alias("rating_stddev"),
        )
        .withColumnRenamed("ml_movie_id", "movie_id")
        .withColumnRenamed("title_clean", "title")
        .withColumnRenamed("genres_final", "genres")
        .filter(col("rating_count") >= 5)
    )


def build_user_features(silver_df):
    """Aggregate user-level features."""
    return (
        silver_df
        .groupBy("ml_user_id")
        .agg(
            count("*").alias("rating_count"),
            avg("ml_rating").alias("avg_rating"),
            spark_min("ml_rating").alias("min_rating"),
            spark_max("ml_rating").alias("max_rating"),
        )
        .withColumnRenamed("ml_user_id", "user_id")
    )


def save_and_register(df, table_name: str, output_path: str, spark: SparkSession):
    path = str(OUTPUT_DIR / output_path)
    df.write.mode("overwrite").parquet(path)
    print(f"  Saved {table_name}: {path}")
    df_reload = spark.read.parquet(path)
    df_reload.createOrReplaceTempView(table_name)
    print(f"  Registered view: {table_name}")


def main():
    spark = create_spark()
    spark.sparkContext.setLogLevel("WARN")

    print("\n=== ETL: Loading bronze layer ===")
    ratings, movies, links, movie_tags = load_movielens(spark)
    load_amazon_ratings(spark)  # logged for dataset size verification; not joined (different ID space)
    tmdb_df = load_tmdb_metadata(spark)

    print("\n=== ETL: Building silver layer ===")
    silver = build_silver_ratings(ratings, movies, links, movie_tags, tmdb_df)
    silver.cache()
    silver_count = silver.count()
    print(f"  Silver ratings: {silver_count:,}")

    print("\n=== ETL: Building gold layer ===")
    gold_ratings = build_gold_als_matrix(silver)
    movie_features = build_movie_features(silver)
    user_features = build_user_features(silver)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== ETL: Saving and registering ===")
    save_and_register(gold_ratings, "gold_ratings", "gold/ratings", spark)
    save_and_register(movie_features, "movie_features", "movie_features", spark)
    save_and_register(user_features, "user_features", "user_features", spark)

    # Also save silver for downstream use
    silver.write.mode("overwrite").parquet(str(OUTPUT_DIR / "silver/ratings"))
    print("  Saved silver layer")

    print("\n=== ETL: Summary ===")
    print(f"  Users: {gold_ratings.select('user_id').distinct().count():,}")
    print(f"  Movies: {gold_ratings.select('movie_id').distinct().count():,}")
    print(f"  Ratings: {gold_ratings.count():,}")
    print(f"  Sparsity: {1 - gold_ratings.count() / (gold_ratings.select('user_id').distinct().count() * gold_ratings.select('movie_id').distinct().count()):.4%}")

    print("\nETL complete.")
    spark.stop()


if __name__ == "__main__":
    main()
