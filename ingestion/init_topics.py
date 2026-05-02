"""Create Kafka topics required by the pipeline."""
import os
import time
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError
from dotenv import load_dotenv

load_dotenv()

BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")

TOPICS = [
    NewTopic(name="user-events", num_partitions=3, replication_factor=1),
    NewTopic(name="rating-updates", num_partitions=1, replication_factor=1),
    NewTopic(name="trending-updates", num_partitions=1, replication_factor=1),
]


def create_topics(retries: int = 10, delay: int = 5):
    for attempt in range(retries):
        try:
            admin = KafkaAdminClient(bootstrap_servers=BOOTSTRAP_SERVERS, client_id="init-topics")
            existing = set(admin.list_topics())
            new_topics = [t for t in TOPICS if t.name not in existing]
            if new_topics:
                admin.create_topics(new_topics=new_topics, validate_only=False)
                print(f"Created topics: {[t.name for t in new_topics]}")
            else:
                print("All topics already exist.")
            admin.close()
            return
        except TopicAlreadyExistsError:
            pass
        except Exception as e:
            print(f"Attempt {attempt + 1}/{retries} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
    print("ERROR: Could not connect to Kafka after retries.")


if __name__ == "__main__":
    create_topics()
