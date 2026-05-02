"""
Download MovieLens 25M and Amazon Reviews (Movies & TV) datasets.

Amazon Reviews 2023: https://amazon-reviews-2023.github.io/
MovieLens 25M: https://grouplens.org/datasets/movielens/25m/
"""
import os
import sys
import zipfile
import subprocess
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path(__file__).parent

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
# Amazon Reviews 2023 - Movies & TV (ratings only, ~1.7GB compressed)
AMAZON_RATINGS_URL = "https://datarepo.eng.ucsd.edu/repo_public/amazon_2023/benchmark/5-core/rating_only/Movies_and_TV.jsonl.gz"
# Full reviews with text (~10GB compressed) — comment out if storage is tight
AMAZON_REVIEWS_URL = "https://datarepo.eng.ucsd.edu/repo_public/amazon_2023/data/review_categories/Movies_and_TV.jsonl.gz"


def download(url: str, dest: Path, skip_if_exists: bool = True):
    if skip_if_exists and dest.exists():
        print(f"  Already exists: {dest} ({dest.stat().st_size / 1e9:.2f} GB)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {url}")
    print(f"  -> {dest}")
    # Use curl: handles SSL certs properly on macOS, shows progress natively
    result = subprocess.run(
        ["curl", "-L", "--progress-bar", "-o", str(dest), url],
        check=True,
    )
    print(f"  Done: {dest.stat().st_size / 1e9:.2f} GB")


def download_movielens():
    print("\n=== MovieLens 25M ===")
    zip_path = DATA_DIR / "movielens" / "ml-25m.zip"
    extract_dir = DATA_DIR / "movielens"

    download(MOVIELENS_URL, zip_path)

    if not (extract_dir / "ml-25m" / "ratings.csv").exists():
        print("  Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        print("  Extracted.")

    ml_dir = extract_dir / "ml-25m"
    for fname in ["ratings.csv", "movies.csv", "tags.csv", "links.csv"]:
        fpath = ml_dir / fname
        if fpath.exists():
            size = fpath.stat().st_size / 1e6
            print(f"  {fname}: {size:.1f} MB")


def download_amazon_reviews(full_reviews: bool = True):
    print("\n=== Amazon Reviews — Movies & TV ===")
    amazon_dir = DATA_DIR / "amazon"
    amazon_dir.mkdir(parents=True, exist_ok=True)

    # Always download ratings (small, needed for ALS)
    ratings_path = amazon_dir / "Movies_and_TV_ratings.jsonl.gz"
    download(AMAZON_RATINGS_URL, ratings_path)

    if full_reviews:
        reviews_path = amazon_dir / "Movies_and_TV.jsonl.gz"
        print("\n  Full reviews (~10GB compressed). Press Ctrl+C to skip.")
        try:
            download(AMAZON_REVIEWS_URL, reviews_path)
        except KeyboardInterrupt:
            print("\n  Skipped full reviews download.")


def verify_sizes():
    print("\n=== Dataset sizes ===")
    total = 0
    for p in DATA_DIR.rglob("*"):
        if p.is_file() and not p.suffix == ".py":
            size = p.stat().st_size
            total += size
            if size > 1e6:
                print(f"  {p.relative_to(DATA_DIR)}: {size / 1e9:.3f} GB")
    print(f"\n  Total: {total / 1e9:.2f} GB")
    if total < 10e9:
        print("  WARNING: Total < 10 GB. Download full Amazon reviews to meet requirement.")
    else:
        print("  10 GB requirement satisfied.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-full-reviews", action="store_true",
                        help="Skip 10GB Amazon reviews, download ratings only")
    args = parser.parse_args()

    download_movielens()
    download_amazon_reviews(full_reviews=not args.no_full_reviews)
    verify_sizes()
