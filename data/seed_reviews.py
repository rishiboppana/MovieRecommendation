"""
Seed the reviews SQLite DB with realistic human-named reviews for every movie.
Reviews are generated deterministically based on movie_id + avg_rating.

Run: python data/seed_reviews.py
"""
import random
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

DATA_DIR  = Path(__file__).parent
REVIEWS_DB = DATA_DIR / "reviews.db"

FIRST_NAMES = [
    "James","Mary","Robert","Patricia","John","Jennifer","Michael","Linda",
    "David","Barbara","William","Elizabeth","Richard","Susan","Joseph","Jessica",
    "Thomas","Sarah","Charles","Karen","Christopher","Lisa","Daniel","Nancy",
    "Matthew","Betty","Anthony","Margaret","Mark","Sandra","Donald","Ashley",
    "Steven","Dorothy","Paul","Kimberly","Andrew","Emily","Joshua","Donna",
    "Kenneth","Michelle","Kevin","Carol","Brian","Amanda","George","Melissa",
    "Timothy","Deborah","Ronald","Stephanie","Edward","Rebecca","Jason","Sharon",
    "Jeffrey","Laura","Ryan","Cynthia","Jacob","Kathleen","Gary","Amy",
    "Nicholas","Angela","Eric","Shirley","Jonathan","Anna","Stephen","Brenda",
    "Larry","Pamela","Justin","Emma","Scott","Nicole","Brandon","Helen",
    "Benjamin","Samantha","Samuel","Katherine","Christine","Rachel","Maria",
    "Heather","Olivia","Noah","Liam","Sophia","Isabella","Ava","Mia","Luna",
    "Ethan","Lucas","Mason","Aiden","Harper","Evelyn","Abigail","Charlotte",
    "Sofia","Elijah","Logan","Jackson","Sebastian","Mateo","Jack","Owen",
    "Theodore","Caleb","Zoe","Nora","Lily","Eleanor","Hannah","Lillian",
    "Addison","Aubrey","Ellie","Stella","Natalie","Zara","Vivian","Aurora",
]

LAST_NAMES = [
    "Smith","Johnson","Williams","Brown","Jones","Garcia","Miller","Davis",
    "Rodriguez","Martinez","Hernandez","Lopez","Gonzalez","Wilson","Anderson",
    "Thomas","Taylor","Moore","Jackson","Martin","Lee","Perez","Thompson",
    "White","Harris","Sanchez","Clark","Ramirez","Lewis","Robinson","Walker",
    "Young","Allen","King","Wright","Scott","Torres","Nguyen","Hill","Flores",
    "Green","Adams","Nelson","Baker","Hall","Rivera","Campbell","Mitchell",
    "Carter","Roberts","Chen","Kim","Park","Patel","Shah","O'Brien","Murphy",
    "Walsh","Collins","Stewart","Reed","Cooper","Bailey","Bell","Cox",
    "Richardson","Ward","Russell","Butler","Hughes","Foster","Coleman","Jenkins",
    "Patterson","Griffin","Diaz","Hayes","Myers","Ford","Hamilton","Graham",
    "Sullivan","Wallace","Woods","Cole","West","Jordan","Owens","Reynolds",
]

REVIEWS = {
    5: [
        "Absolutely masterful. One of the best films I've ever seen.",
        "A cinematic triumph. The performances are extraordinary and the story is unforgettable.",
        "I've watched this three times and it keeps getting better. A true masterpiece.",
        "Everything about this film works perfectly — direction, acting, story. Top notch.",
        "This changed the way I think about cinema. A must-watch for everyone.",
        "Stunning. I was completely absorbed from the first frame to the last.",
        "One of those rare films that stays with you long after the credits roll.",
        "Flawless storytelling. Every scene serves a purpose and the payoff is tremendous.",
        "Genuinely moved by this film. The performances are some of the best I've ever seen.",
        "Cannot recommend this enough. A landmark piece of filmmaking.",
        "The cinematography alone is worth the watch. Absolutely breathtaking.",
        "Watched it with my whole family and everyone was riveted. A perfect film.",
        "I don't say this lightly but this is a once-in-a-generation kind of movie.",
        "The ending hit me so hard I had to sit in silence for a few minutes. Incredible.",
        "Every element works in perfect harmony. Pure cinema.",
    ],
    4: [
        "Really enjoyable film with strong performances and a compelling story.",
        "A solid film that delivers on its promises. Well worth watching.",
        "Thoroughly entertaining from start to finish. Minor flaws but overall excellent.",
        "Great performances carry this film to new heights. Highly recommend.",
        "One of the better films in its genre. Engaging and well-crafted.",
        "Very satisfying watch. The characters are well-developed and the story moves well.",
        "Strong direction and a great script. Kept me hooked throughout.",
        "Impressive on multiple levels. The third act is especially fantastic.",
        "A film that knows exactly what it wants to be and delivers it with style.",
        "I went in with low expectations and was pleasantly surprised. Really good.",
        "The performances are what elevate this above average. Great cast all around.",
        "Solid entertainment that respects the audience's intelligence.",
        "Visually gorgeous and emotionally resonant. A very complete film.",
        "Not without its flaws but the strengths far outweigh them. Worth your time.",
        "Smart, funny, and surprisingly moving in places. Really enjoyed this one.",
    ],
    3: [
        "A decent film but nothing particularly special. Worth a watch if you like the genre.",
        "Some good moments but the pacing drags in the second half.",
        "It's fine. Competently made but doesn't leave much of an impression.",
        "Mixed feelings on this one. The first half is stronger than the second.",
        "Average fare for the genre. Enjoyable enough but forgettable.",
        "Had some great scenes but the overall package is just okay.",
        "The premise is interesting but the execution is a little flat.",
        "Not bad, not great. A perfectly acceptable way to spend two hours.",
        "I enjoyed parts of it but the story loses its way toward the end.",
        "Decent enough. The cast is clearly trying hard with so-so material.",
        "Middle-of-the-road filmmaking. Watchable but nothing to get excited about.",
        "It has its moments but struggles to maintain momentum throughout.",
        "The setup is strong but the payoff doesn't quite deliver.",
        "Enjoyable in places, frustrating in others. A mixed bag overall.",
    ],
    2: [
        "Had potential but didn't deliver. The story feels rushed and underdeveloped.",
        "Disappointing considering the talent involved. Expected much more.",
        "A few good scenes but overall the film fails to come together.",
        "The premise was promising but the execution left much to be desired.",
        "I kept waiting for it to get interesting. It never quite did.",
        "Shallow and formulaic. By-the-numbers filmmaking with nothing new to say.",
        "The characters are frustratingly one-dimensional. Hard to care about anyone.",
        "Started okay but fell apart in the second half. Very disappointing.",
        "I don't understand the positive reviews. Nothing here stands out.",
        "Not enough here to justify the runtime. Could have been 30 minutes shorter.",
        "The editing is choppy and the story doesn't flow naturally.",
        "Feels like a first draft. A lot of ideas that never fully develop.",
    ],
    1: [
        "Couldn't make it past the first hour. Just not for me.",
        "A real disappointment. The story is a mess and the characters are unlikable.",
        "Genuinely tedious. Save your time for something better.",
        "One of the worst films I've seen in years. Everything feels off.",
        "Painful to sit through. Confusing, poorly paced, and completely forgettable.",
        "The writing is atrocious. Characters make decisions that defy all logic.",
        "I want those two hours back. Nothing redeemable here.",
        "Poorly directed and poorly acted. A waste of a promising premise.",
    ],
}


def star_bucket(avg_rating: float) -> int:
    """Map avg_rating to a 1-5 star bucket, weighted realistically."""
    if avg_rating >= 4.2:
        return random.choices([5, 4, 3], weights=[65, 30, 5])[0]
    elif avg_rating >= 3.5:
        return random.choices([5, 4, 3, 2], weights=[15, 45, 35, 5])[0]
    elif avg_rating >= 2.8:
        return random.choices([4, 3, 2, 1], weights=[10, 45, 35, 10])[0]
    elif avg_rating >= 2.0:
        return random.choices([3, 2, 1], weights=[20, 45, 35])[0]
    else:
        return random.choices([2, 1], weights=[35, 65])[0]


def make_name(rng: random.Random) -> str:
    return f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"


def random_date(rng: random.Random) -> str:
    days_ago = rng.randint(1, 1460)  # up to 4 years back
    return (datetime.utcnow() - timedelta(days=days_ago)).isoformat()


def main():
    path = DATA_DIR / "processed" / "movie_features"
    if not path.exists():
        print("ERROR: movie_features not found — run ETL first")
        return

    df = pq.read_table(str(path), columns=["movie_id", "avg_rating"]).to_pandas()
    df = df.dropna(subset=["avg_rating"])
    print(f"Seeding reviews for {len(df):,} movies…")

    conn = sqlite3.connect(str(REVIEWS_DB))
    # Ensure schema has username column
    try:
        conn.execute("ALTER TABLE reviews ADD COLUMN username TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        pass

    rows = []
    for _, row in df.iterrows():
        movie_id = int(row["movie_id"])
        avg_r    = float(row["avg_rating"])
        rng      = random.Random(movie_id * 31337)  # deterministic per movie

        n = rng.randint(3, 6)
        for i in range(n):
            stars     = star_bucket(avg_r)
            text      = rng.choice(REVIEWS[stars])
            username  = make_name(rng)
            created   = random_date(rng)
            # Synthetic user_id: unique by construction (movie_id * 10 + i)
            uid       = 200_001 + movie_id * 10 + i
            rows.append((uid, movie_id, float(stars), text, username, created))

    conn.executemany("""
        INSERT OR IGNORE INTO reviews
            (user_id, movie_id, rating, review, username, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, rows)
    conn.commit()
    count = conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
    conn.close()
    print(f"Done — {len(rows):,} rows inserted, {count:,} total reviews in DB")


if __name__ == "__main__":
    main()
