"""
Simulate user click/rating events and publish to Kafka topic 'user-events'.

Generates realistic event streams:
  - Users browse (click/view events)
  - Users rate movies (rate events)
  - Users skip movies (skip events)

Usage:
    python ingestion/kafka_producer.py --rate 100 --duration 300
"""
import os
import json
import time
import random
import argparse
from datetime import datetime, timezone
from kafka import KafkaProducer
from dotenv import load_dotenv

load_dotenv()

BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")
TOPIC = "user-events"

EVENT_TYPES = ["click", "view", "rate", "skip"]
EVENT_WEIGHTS = [0.35, 0.30, 0.20, 0.15]

# Realistic rating distribution (skewed high like Netflix/Amazon)
RATING_WEIGHTS = [0.05, 0.08, 0.15, 0.30, 0.42]  # 1..5
RATINGS = [1.0, 2.0, 3.0, 4.0, 5.0]


def make_event(user_pool: list[int], movie_pool: list[int]) -> dict:
    event_type = random.choices(EVENT_TYPES, weights=EVENT_WEIGHTS)[0]
    return {
        "user_id": random.choice(user_pool),
        "movie_id": random.choice(movie_pool),
        "event_type": event_type,
        "rating": random.choices(RATINGS, weights=RATING_WEIGHTS)[0] if event_type == "rate" else None,
        "session_id": random.randint(100000, 999999),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def main(events_per_sec: int = 10, duration_sec: int | None = None):
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: str(k).encode("utf-8"),
        linger_ms=10,
        batch_size=32768,
        compression_type="gzip",
    )

    # Simulate 10k active users, 60k movies (MovieLens-25M scale)
    user_pool = random.sample(range(1, 162542), k=min(10000, 162541))
    movie_pool = random.sample(range(1, 209172), k=min(60000, 209171))

    print(f"Publishing to {TOPIC} @ {BOOTSTRAP_SERVERS}")
    print(f"Rate: {events_per_sec} events/sec | Duration: {duration_sec or 'infinite'} sec")
    print("Press Ctrl+C to stop\n")

    interval = 1.0 / events_per_sec
    total_sent = 0
    start = time.time()

    try:
        while True:
            if duration_sec and (time.time() - start) >= duration_sec:
                break

            event = make_event(user_pool, movie_pool)
            producer.send(
                TOPIC,
                key=event["user_id"],
                value=event,
            )
            total_sent += 1

            if total_sent % 1000 == 0:
                elapsed = time.time() - start
                print(f"  Sent {total_sent:,} events in {elapsed:.1f}s ({total_sent/elapsed:.0f} eps)")

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        producer.flush()
        producer.close()
        elapsed = time.time() - start
        print(f"\nTotal: {total_sent:,} events in {elapsed:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate", type=int, default=10, help="Events per second (default: 10)")
    parser.add_argument("--duration", type=int, default=None, help="Duration in seconds (default: infinite)")
    args = parser.parse_args()
    main(events_per_sec=args.rate, duration_sec=args.duration)
