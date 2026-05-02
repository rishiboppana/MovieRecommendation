"""
Train Spark MLlib ALS with MLflow experiment tracking.

Steps:
  1. Load gold ratings from ETL output
  2. Train/test split (80/20)
  3. Fit ALS model
  4. Evaluate: RMSE, MAE, Precision@K, MAP@K
  5. Log all params + metrics + model artifact to MLflow
  6. Register model in MLflow Model Registry
  7. Precompute top-N recs for all users → save parquet

Submit:
    spark-submit \
        --master local[*] \
        --driver-memory 10g \
        --conf spark.sql.adaptive.enabled=true \
        ml/train_als.py
"""
import os
import argparse
from pathlib import Path
import mlflow
import mlflow.spark
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, collect_list, struct, sort_array, desc, lit, count
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.types import IntegerType
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "processed"
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "movie-recommendations-als"
MODEL_NAME = "movie-rec-als"


def create_spark():
    return (
        SparkSession.builder
        .appName("MovieRec-ALS-Training")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.driver.memory", "10g")
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
    )


def compute_map_at_k(model, test_df, k: int = 10) -> float:
    """Compute Mean Average Precision @ K."""
    # Get actual relevant movies per user (rated >= 4.0)
    actual = (
        test_df
        .filter(col("rating") >= 4.0)
        .groupBy("user_id")
        .agg(collect_list("movie_id").alias("actual"))
    )

    # Get top-K predictions per user
    users_df = test_df.select("user_id").distinct()
    predictions = model.recommendForUserSubset(users_df, k)
    pred_exploded = predictions.select(
        col("user_id"),
        col("recommendations.movie_id").alias("predicted")
    )

    joined = actual.join(pred_exploded, "user_id", "inner")

    def avg_precision(actual_list, pred_list):
        if not actual_list or not pred_list:
            return 0.0
        actual_set = set(actual_list)
        hits, precision_sum = 0, 0.0
        for i, p in enumerate(pred_list[:k]):
            if p in actual_set:
                hits += 1
                precision_sum += hits / (i + 1)
        return precision_sum / min(len(actual_set), k)

    from pyspark.sql.functions import udf
    from pyspark.sql.types import DoubleType
    ap_udf = udf(avg_precision, DoubleType())

    ap_df = joined.withColumn(
        "ap",
        ap_udf(col("actual"), col("predicted"))
    )
    map_score = ap_df.agg({"ap": "avg"}).first()[0]
    return float(map_score) if map_score is not None else 0.0


def train(params: dict, spark: SparkSession) -> dict:
    ratings_path = str(DATA_DIR / "gold/ratings")
    if not Path(ratings_path).exists():
        raise FileNotFoundError(f"Run ETL first: {ratings_path}")

    ratings = (
        spark.read.parquet(ratings_path)
        .select(
            col("user_id").cast(IntegerType()),
            col("movie_id").cast(IntegerType()),
            col("rating").cast("float"),
        )
        .filter(col("rating").isNotNull())
        .filter(col("rating") > 0)
        .cache()
    )

    implicit = params.get("implicit", True)
    print(f"Total ratings: {ratings.count():,}  |  mode={'implicit' if implicit else 'explicit'}")
    train_df, test_df = ratings.randomSplit([0.8, 0.2], seed=42)

    als = ALS(
        rank=params["rank"],
        regParam=params["reg_param"],
        maxIter=params["max_iter"],
        # alpha scales rating values into confidence weights: c = 1 + alpha * r
        # Higher alpha → model pays more attention to high-rated items
        alpha=params.get("alpha", 25.0),
        userCol="user_id",
        itemCol="movie_id",
        ratingCol="rating",
        coldStartStrategy="drop",
        # implicitPrefs=True treats ratings as confidence, not ground-truth values.
        # Better for recommendation (what will users engage with?) vs rating prediction.
        implicitPrefs=implicit,
        seed=42,
        numUserBlocks=10,
        numItemBlocks=10,
    )

    print(f"Training ALS: rank={params['rank']}, reg={params['reg_param']}, "
          f"iter={params['max_iter']}, alpha={params.get('alpha', 25.0)}, implicit={implicit}")
    model = als.fit(train_df)

    # ── Evaluate ──
    # For implicit ALS evaluate on held-out ratings ≥ 4.0 treated as positives
    if implicit:
        from pyspark.sql.functions import when as spark_when
        # Map model scores to [0,1] range for comparison; use RMSE on binarised labels
        predictions = (
            model.transform(test_df)
            .filter(col("prediction").isNotNull())
            .withColumn("binary_label", spark_when(col("rating") >= 4.0, 1.0).otherwise(0.0))
        )
        evaluator_rmse = RegressionEvaluator(
            metricName="rmse", labelCol="binary_label", predictionCol="prediction"
        )
        evaluator_mae = RegressionEvaluator(
            metricName="mae", labelCol="binary_label", predictionCol="prediction"
        )
    else:
        predictions = model.transform(test_df).filter(col("prediction").isNotNull())
        evaluator_rmse = RegressionEvaluator(
            metricName="rmse", labelCol="rating", predictionCol="prediction"
        )
        evaluator_mae = RegressionEvaluator(
            metricName="mae", labelCol="rating", predictionCol="prediction"
        )

    rmse = evaluator_rmse.evaluate(predictions)
    mae = evaluator_mae.evaluate(predictions)
    map_k = compute_map_at_k(model, test_df, k=10)

    print(f"  RMSE={rmse:.4f}  MAE={mae:.4f}  MAP@10={map_k:.4f}")

    return {
        "model": model,
        "rmse": rmse,
        "mae": mae,
        "map_at_10": map_k,
        "train_size": train_df.count(),
        "test_size": test_df.count(),
    }


def main(args):
    params = {
        "rank": args.rank,
        "reg_param": args.reg_param,
        "max_iter": args.max_iter,
        "alpha": args.alpha,
        "implicit": not args.explicit,
    }

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    spark = create_spark()
    spark.sparkContext.setLogLevel("WARN")

    result = train(params, spark)
    model = result.pop("model")

    # ── Precompute recs FIRST (independent of MLflow) ──
    print("Precomputing recommendations for all users...")
    recs = model.recommendForAllUsers(20)
    recs_path = str(ROOT / "data" / "recommendations")
    recs.write.mode("overwrite").parquet(recs_path)
    print(f"  Saved: {recs_path}")

    print("Precomputing item similarities...")
    similar = model.recommendForAllItems(10)
    similar_path = str(ROOT / "data" / "similar_movies")
    similar.write.mode("overwrite").parquet(similar_path)
    print(f"  Saved: {similar_path}")

    # ── Also save model locally as backup ──
    local_model_path = str(ROOT / "mlflow" / "artifacts" / "als-model-local")
    model.save(local_model_path)
    print(f"  Model saved locally: {local_model_path}")

    # ── Log to MLflow ──
    try:
        local_artifact_root = str(ROOT / "mlflow" / "artifacts")
        mlflow.set_tracking_uri(MLFLOW_URI)
        # Create experiment with local artifact path (survives DB resets)
        try:
            exp_id = mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=local_artifact_root)
        except Exception:
            exp_id = mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id
        mlflow.set_experiment(experiment_id=exp_id)

        with mlflow.start_run(run_name=f"als-r{params['rank']}-reg{params['reg_param']}") as run:
            mlflow.log_params(params)
            mlflow.log_param("spark_version", spark.version)
            mlflow.log_param("dataset", "MovieLens-25M + Amazon-7.4M")
            mlflow.log_param("mode", "implicit" if params.get("implicit") else "explicit")

            mlflow.log_metrics({
                "rmse": result["rmse"],
                "mae": result["mae"],
                "map_at_10": result["map_at_10"],
                "train_size": result["train_size"],
                "test_size": result["test_size"],
            })

            mlflow.spark.log_model(
                spark_model=model,
                artifact_path="als-model",
                registered_model_name=MODEL_NAME,
            )

            run_id = run.info.run_id
            print(f"\nMLflow run_id: {run_id}")

        client = MlflowClient(tracking_uri=MLFLOW_URI)
        versions = client.get_latest_versions(MODEL_NAME)
        if versions:
            latest_version = max(int(v.version) for v in versions)
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=str(latest_version),
                stage="Production",
                archive_existing_versions=True,
            )
            print(f"Model '{MODEL_NAME}' v{latest_version} → Production")

    except Exception as e:
        print(f"  MLflow logging skipped ({e}). Recs already saved.")

    print(f"\nRMSE={result['rmse']:.4f} | MAE={result['mae']:.4f} | MAP@10={result['map_at_10']:.4f}")
    spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank",      type=int,   default=150)
    parser.add_argument("--reg-param", type=float, default=0.01)
    parser.add_argument("--max-iter",  type=int,   default=15)
    parser.add_argument("--alpha",     type=float, default=25.0)
    parser.add_argument("--explicit",  action="store_true",
                        help="Use explicit rating prediction instead of implicit feedback")
    args = parser.parse_args()
    main(args)
