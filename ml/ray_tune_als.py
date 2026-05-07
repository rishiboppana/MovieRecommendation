"""
Distributed hyperparameter search for ALS using Ray Tune.

Search space:
  - rank: [10, 50, 100, 150, 200]
  - reg_param: [0.01, 0.05, 0.1, 0.3, 0.5, 1.0]
  - max_iter: [5, 10, 15, 20]

Uses ASHAScheduler for early stopping (prune bad configs after 5 iters).
Best config is re-trained with full iters and logged to MLflow.

Usage:
    python ml/ray_tune_als.py --num-samples 20 --max-concurrent 4
"""
import os
import sys
import json
import argparse
import tempfile
from pathlib import Path
import mlflow
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "processed"
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
EXPERIMENT_NAME = "movie-recommendations-als"


def als_trial(config: dict):
    """Ray Tune trial function — creates a local Spark session per trial."""
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col
    from pyspark.sql.types import IntegerType
    from pyspark.ml.recommendation import ALS
    from pyspark.ml.evaluation import RegressionEvaluator

    spark = (
        SparkSession.builder
        .appName(f"ALS-Trial-r{config['rank']}")
        .master("local[2]")
        .config("spark.driver.memory", "4g")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "50")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    try:
        ratings_path = str(DATA_DIR / "gold/ratings")
        ratings = (
            spark.read.parquet(ratings_path)
            .select(
                col("user_id").cast(IntegerType()),
                col("movie_id").cast(IntegerType()),
                col("rating").cast("float"),
            )
            .filter(col("rating").isNotNull())
            # Sample for faster HPO trials
            .sample(fraction=0.3, seed=42)
        )

        train_df, val_df = ratings.randomSplit([0.8, 0.2], seed=42)

        als = ALS(
            rank=config["rank"],
            regParam=config["reg_param"],
            maxIter=config["max_iter"],
            userCol="user_id",
            itemCol="movie_id",
            ratingCol="rating",
            coldStartStrategy="drop",
            seed=42,
        )

        model = als.fit(train_df)
        preds = model.transform(val_df).filter(col("prediction").isNotNull())

        evaluator = RegressionEvaluator(
            metricName="rmse", labelCol="rating", predictionCol="prediction"
        )
        rmse = evaluator.evaluate(preds)
        ray.train.report({"rmse": rmse, "rank": config["rank"]})

    finally:
        spark.stop()


def main(num_samples: int = 20, max_concurrent: int = 2):
    if not (DATA_DIR / "gold/ratings").exists():
        print("ERROR: Run ETL first (make etl)")
        sys.exit(1)

    ray.init(ignore_reinit_error=True, num_cpus=max(4, max_concurrent * 2))

    search_space = {
        "rank": tune.choice([10, 50, 100, 150, 200]),
        "reg_param": tune.loguniform(0.01, 1.0),
        "max_iter": tune.choice([5, 10, 15, 20]),
    }

    scheduler = ASHAScheduler(
        metric="rmse",
        mode="min",
        max_t=20,
        grace_period=5,
        reduction_factor=2,
    )

    search_alg = OptunaSearch(metric="rmse", mode="min")

    tuner = tune.Tuner(
        als_trial,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            num_samples=num_samples,
            max_concurrent_trials=max_concurrent,
            scheduler=scheduler,
            search_alg=search_alg,
        ),
        run_config=ray.train.RunConfig(
            name="als-hpo",
            storage_path=str(ROOT / "ray_results"),
            verbose=1,
        ),
    )

    results = tuner.fit()
    best = results.get_best_result(metric="rmse", mode="min")
    best_config = best.config

    print("\n=== Best Hyperparameters ===")
    print(json.dumps(best_config, indent=2))
    print(f"Best RMSE: {best.metrics['rmse']:.4f}")

    # Save best config
    best_config_path = ROOT / "ml" / "best_config.json"
    with open(best_config_path, "w") as f:
        json.dump({**best_config, "best_rmse": best.metrics["rmse"]}, f, indent=2)
    print(f"Saved to {best_config_path}")

    print("\nRe-training best config with full iterations...")
    os.system(
        f"cd {ROOT} && /opt/spark/bin/spark-submit "
        f"--master local[*] --driver-memory 10g "
        f"ml/train_als.py "
        f"--rank {best_config['rank']} "
        f"--reg-param {best_config['reg_param']:.4f} "
        f"--max-iter {best_config['max_iter']}"
    )

    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--max-concurrent", type=int, default=2)
    args = parser.parse_args()
    main(num_samples=args.num_samples, max_concurrent=args.max_concurrent)
