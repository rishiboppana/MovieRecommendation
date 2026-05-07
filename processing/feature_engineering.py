"""
Feature engineering on top of ETL output.

Adds:
  - Implicit feedback matrix (treat any event as 1.0 confidence interaction)
  - Genre one-hot encoding (19 genres)
  - Genome tag relevance vectors (1,128-dim content features per movie)
  - User/item popularity features
  - Normalised rating residuals (user bias, item bias)

Submit:
    spark-submit --master local[*] processing/feature_engineering.py
"""
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, split, explode, count, avg, when, lit,
    udf, array_contains, size, desc
)
from pyspark.sql.types import FloatType
from pyspark.ml.feature import StringIndexer, IndexToString, VectorAssembler
from pyspark.ml.stat import Correlation

ROOT        = Path(__file__).parent.parent
DATA_DIR    = ROOT / "data" / "processed"
MOVIELENS_DIR = ROOT / "data" / "movielens" / "ml-25m"

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


def build_genome_tag_vectors(spark):
    """
    Build per-movie genome tag relevance vectors from genome-scores.csv.

    genome-scores.csv: 15,292,353 rows — every (movie, tag) pair with a
    relevance score in [0, 1]. We keep the top-20 tags per movie (by
    relevance) and pivot them into a wide feature table. This gives a
    content-based representation that complements ALS collaborative features.

    Output schema:
        movie_id  INT
        tag_*     DOUBLE  (one column per top-20 tag, relevance score)
        top_tags  STRING  (pipe-separated top-5 tag names, for display)
    """
    scores_path = str(MOVIELENS_DIR / "genome-scores.csv")
    tags_path   = str(MOVIELENS_DIR / "genome-tags.csv")

    if not Path(scores_path).exists():
        print("  genome-scores.csv not found — skipping genome feature engineering.")
        return None

    print("Loading genome scores for feature engineering (15M rows)...")
    genome_scores = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(scores_path)
    )
    genome_tags = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(tags_path)
    )

    # Join to get tag names
    scored = genome_scores.join(genome_tags, "tagId", "inner")

    # Top-5 tags per movie for the display column (genome_themes)
    from pyspark.sql.window import Window
    from pyspark.sql.functions import row_number, concat_ws, collect_list
    w = Window.partitionBy("movieId").orderBy(desc("relevance"))
    top5 = (
        scored
        .filter(col("relevance") >= 0.5)
        .withColumn("rn", row_number().over(w))
        .filter(col("rn") <= 5)
        .groupBy("movieId")
        .agg(concat_ws("|", collect_list("tag")).alias("top_tags"))
        .withColumnRenamed("movieId", "movie_id")
    )

    # Top-20 tag relevance scores per movie as wide features
    top20_tags = (
        scored
        .withColumn("rn", row_number().over(w))
        .filter(col("rn") <= 20)
        .select(
            col("movieId").alias("movie_id"),
            col("tag"),
            col("relevance"),
        )
    )

    # Pivot: one column per unique tag in the top-20 set across all movies
    tag_list = [
        r["tag"] for r in
        top20_tags.select("tag").distinct().orderBy("tag").limit(50).collect()
    ]
    genome_wide = (
        top20_tags
        .filter(col("tag").isin(tag_list))
        .groupBy("movie_id")
        .pivot("tag", tag_list)
        .agg(avg("relevance"))
        .na.fill(0.0)
    )

    result = genome_wide.join(top5, "movie_id", "left")
    coverage = result.count()
    print(f"  Genome tag feature matrix: {coverage:,} movies × {len(tag_list)} tag features")
    return result


def main():
    spark = create_spark()
    spark.sparkContext.setLogLevel("WARN")

    ratings = spark.read.parquet(str(DATA_DIR / "gold/ratings"))
    movies  = spark.read.parquet(str(DATA_DIR / "movie_features"))

    print("Computing rating biases...")
    ratings_biased, global_mean, user_bias, item_bias = compute_biases(ratings)
    print(f"  Global mean: {global_mean:.3f}")

    print("Building implicit confidence matrix...")
    ratings_implicit = build_implicit_confidence(ratings)

    print("Adding genre features...")
    movies_fe = add_genre_features(movies)

    print("Building genome tag feature vectors (15M rows)...")
    genome_vectors = build_genome_tag_vectors(spark)

    out = DATA_DIR / "features"
    out.mkdir(parents=True, exist_ok=True)

    ratings_biased.write.mode("overwrite").parquet(str(out / "ratings_biased"))
    ratings_implicit.write.mode("overwrite").parquet(str(out / "ratings_implicit"))
    movies_fe.write.mode("overwrite").parquet(str(out / "movie_features_fe"))
    user_bias.write.mode("overwrite").parquet(str(out / "user_bias"))
    item_bias.write.mode("overwrite").parquet(str(out / "item_bias"))
    if genome_vectors is not None:
        genome_vectors.write.mode("overwrite").parquet(str(out / "genome_tag_vectors"))
        print(f"  Saved genome_tag_vectors → {out / 'genome_tag_vectors'}")

    print(f"Feature engineering complete → {out}")
    spark.stop()


if __name__ == "__main__":
    main()
