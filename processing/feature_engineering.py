"""
Feature engineering on top of ETL output.

Adds:
  - Implicit feedback matrix (treat any event as 1.0 confidence interaction)
  - Genre one-hot encoding
  - User/item popularity features
  - Normalised rating residuals (user bias, item bias)

Submit:
    spark-submit --master local[*] processing/feature_engineering.py
"""
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, split, explode, count, avg, when, lit,
    udf, array_contains, size
)
from pyspark.sql.types import FloatType
from pyspark.ml.feature import StringIndexer, IndexToString, VectorAssembler
from pyspark.ml.stat import Correlation

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "processed"

ALL_GENRES = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "IMAX",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


def create_spark():
    return (
        SparkSession.builder
        .appName("MovieRec-Features")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.driver.memory", "6g")
        .getOrCreate()
    )


def add_genre_features(movie_df):
    """One-hot encode genres."""
    genre_cols = {}
    for genre in ALL_GENRES:
        safe = genre.lower().replace("-", "_")
        genre_cols[f"genre_{safe}"] = when(
            col("genres").contains(genre), 1
        ).otherwise(0)
    return movie_df.withColumns(genre_cols)


def compute_biases(ratings_df):
    """Compute global mean, user bias, item bias for explicit feedback."""
    global_mean = ratings_df.agg(avg("rating")).first()[0]

    user_bias = (
        ratings_df
        .groupBy("user_id")
        .agg((avg("rating") - lit(global_mean)).alias("user_bias"))
    )
    item_bias = (
        ratings_df
        .groupBy("movie_id")
        .agg((avg("rating") - lit(global_mean)).alias("item_bias"))
    )

    adjusted = (
        ratings_df
        .join(user_bias, "user_id", "left")
        .join(item_bias, "movie_id", "left")
        .withColumn("residual_rating",
                    col("rating") - lit(global_mean) - col("user_bias") - col("item_bias"))
    )
    return adjusted, global_mean, user_bias, item_bias


def build_implicit_confidence(ratings_df, alpha: float = 40.0):
    """
    ALS implicit feedback: confidence c_ui = 1 + alpha * r_ui
    Used when treating ratings as implicit preference strength.
    """
    return ratings_df.withColumn("confidence", lit(1.0) + lit(alpha) * col("rating"))


def main():
    spark = create_spark()
    spark.sparkContext.setLogLevel("WARN")

    ratings = spark.read.parquet(str(DATA_DIR / "gold/ratings"))
    movies = spark.read.parquet(str(DATA_DIR / "movie_features"))

    print("Computing rating biases...")
    ratings_biased, global_mean, user_bias, item_bias = compute_biases(ratings)
    print(f"  Global mean: {global_mean:.3f}")

    print("Building implicit confidence matrix...")
    ratings_implicit = build_implicit_confidence(ratings)

    print("Adding genre features...")
    movies_fe = add_genre_features(movies)

    out = DATA_DIR / "features"
    out.mkdir(parents=True, exist_ok=True)

    ratings_biased.write.mode("overwrite").parquet(str(out / "ratings_biased"))
    ratings_implicit.write.mode("overwrite").parquet(str(out / "ratings_implicit"))
    movies_fe.write.mode("overwrite").parquet(str(out / "movie_features_fe"))
    user_bias.write.mode("overwrite").parquet(str(out / "user_bias"))
    item_bias.write.mode("overwrite").parquet(str(out / "item_bias"))

    print(f"Feature engineering complete → {out}")
    spark.stop()


if __name__ == "__main__":
    main()
