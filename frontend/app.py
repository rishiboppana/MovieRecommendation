"""
CineRec — Netflix-style Movie Recommendation UI
Model: Spark MLlib ALS trained on 25M ratings (MovieLens 25M + Amazon 7.4M)
"""
import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="CineRec",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Session state ──
if "liked_movies" not in st.session_state:
    st.session_state.liked_movies = {}
if "user_id" not in st.session_state:
    st.session_state.user_id = 1
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "home"

# ─────────────────────────── Styles ───────────────────────────
st.markdown("""
<style>
/* ---------- Base ---------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"],
[data-testid="stMain"], .main .block-container {
    background-color: #0f0f0f !important;
    color: #e8e8e8 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
[data-testid="stSidebar"]       { display: none !important; }
[data-testid="stHeader"]        { background: transparent !important; height: 0 !important; }
[data-testid="stToolbar"],
footer, #MainMenu               { display: none !important; }
.block-container                { padding: 0 2.5rem 5rem !important;
                                  max-width: 100% !important; }
* { box-sizing: border-box; }
::-webkit-scrollbar             { height: 4px; width: 4px; background: #1a1a1a; }
::-webkit-scrollbar-thumb       { background: #444; border-radius: 4px; }

/* ---------- Top nav ---------- */
.nav-bar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 1.2rem 0 1rem;
    border-bottom: 1px solid #1e1e1e;
    margin-bottom: 0.5rem;
}
.nav-logo {
    font-size: 1.5rem; font-weight: 700; letter-spacing: -0.5px;
    color: #e50914;
}
.nav-links {
    display: flex; gap: 2rem; font-size: 0.85rem; color: #aaa;
}
.nav-links a { color: inherit; text-decoration: none; }
.nav-links a:hover { color: #fff; }

/* ---------- Hero ---------- */
.hero-wrap {
    position: relative; height: 480px;
    margin: 0 -2.5rem 0; overflow: hidden; border-radius: 0;
}
.hero-img {
    width: 100%; height: 100%; object-fit: cover; object-position: center 20%;
    filter: brightness(0.45);
}
.hero-grad {
    position: absolute; inset: 0;
    background: linear-gradient(
        to right, rgba(15,15,15,0.95) 0%, rgba(15,15,15,0.5) 50%, transparent 100%
    ), linear-gradient(
        to top, rgba(15,15,15,1) 0%, transparent 40%
    );
}
.hero-body {
    position: absolute; bottom: 14%; left: 4%;
    max-width: 40%;
}
.hero-title  { font-size: 2.8rem; font-weight: 700; line-height: 1.1;
               letter-spacing: -0.5px; margin-bottom: 0.6rem; color: #fff; }
.hero-meta   { font-size: 0.82rem; color: #888; margin-bottom: 0.7rem; letter-spacing: 0.3px; }
.hero-desc   { font-size: 0.95rem; color: #ccc; line-height: 1.6; margin-bottom: 1.4rem;
               display: -webkit-box; -webkit-line-clamp: 3;
               -webkit-box-orient: vertical; overflow: hidden; }
.hero-actions { display: flex; gap: 0.8rem; }
.btn-play {
    padding: 0.6rem 1.8rem; background: #fff; color: #000;
    border: none; border-radius: 4px; font-size: 0.9rem; font-weight: 600;
    cursor: pointer; letter-spacing: 0.2px;
}
.btn-more {
    padding: 0.6rem 1.6rem; background: rgba(109,109,110,0.5); color: #fff;
    border: none; border-radius: 4px; font-size: 0.9rem; font-weight: 500;
    cursor: pointer; letter-spacing: 0.2px; backdrop-filter: blur(4px);
}

/* ---------- Section titles ---------- */
.section-title {
    font-size: 1.1rem; font-weight: 600; color: #e8e8e8;
    margin: 2.2rem 0 0.8rem; letter-spacing: 0.1px;
}
.section-title.accent { color: #e50914; }
.section-title .count {
    font-size: 0.78rem; font-weight: 400; color: #666; margin-left: 0.6rem;
}
.taste-tag {
    display: inline-block; font-size: 0.68rem; font-weight: 500;
    background: #e50914; color: #fff; padding: 2px 7px; border-radius: 3px;
    margin-left: 0.5rem; vertical-align: middle; letter-spacing: 0.3px;
}

/* ---------- Movie card ---------- */
.card-outer {
    position: relative;
    border-radius: 4px; overflow: visible;
}
.card-inner {
    border-radius: 4px; overflow: hidden;
    aspect-ratio: 2/3;
    background: #1c1c1c;
    transition: transform 0.22s cubic-bezier(.25,.46,.45,.94),
                box-shadow 0.22s ease;
    cursor: pointer;
    position: relative;
}
.card-inner:hover {
    transform: scale(1.07) translateY(-4px);
    box-shadow: 0 16px 40px rgba(0,0,0,0.85);
    z-index: 20;
}
.card-inner img { width: 100%; height: 100%; object-fit: cover; display: block; }
.card-no-poster {
    width: 100%; height: 100%;
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; padding: 1rem; text-align: center;
    background: linear-gradient(135deg, #1c1c1c, #252525);
}
.card-no-poster .np-title {
    font-size: 0.78rem; font-weight: 600; color: #ccc;
    line-height: 1.3; margin-top: 0.4rem;
}
.card-hover-overlay {
    position: absolute; inset: 0;
    background: linear-gradient(transparent 45%, rgba(0,0,0,0.92) 100%);
    opacity: 0; transition: opacity 0.22s ease;
    display: flex; flex-direction: column; justify-content: flex-end;
    padding: 0.7rem 0.6rem 0.5rem;
    border-radius: 4px;
}
.card-inner:hover .card-hover-overlay { opacity: 1; }
.card-title-ov  { font-size: 0.72rem; font-weight: 600; color: #fff;
                  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.card-genre-ov  { font-size: 0.62rem; color: #aaa; margin-top: 2px; }
.liked-dot {
    position: absolute; top: 6px; right: 6px; width: 8px; height: 8px;
    background: #e50914; border-radius: 50%; z-index: 5;
    box-shadow: 0 0 6px #e50914;
}
.liked-outline { outline: 2px solid #e50914; outline-offset: 1px; }

/* ---------- Like button ---------- */
.stButton > button {
    background: transparent !important;
    border: 1px solid #333 !important;
    color: #999 !important;
    border-radius: 3px !important;
    font-size: 0.7rem !important;
    padding: 3px 0 !important;
    width: 100% !important;
    transition: all 0.15s ease !important;
    letter-spacing: 0.2px !important;
}
.stButton > button:hover {
    border-color: #666 !important;
    color: #fff !important;
    background: rgba(255,255,255,0.05) !important;
}

/* ---------- Search ---------- */
[data-testid="stTextInput"] input {
    background: #1c1c1c !important; border: 1px solid #2e2e2e !important;
    color: #e8e8e8 !important; border-radius: 4px !important;
    font-size: 0.88rem !important; padding: 0.55rem 1rem !important;
    transition: border-color 0.2s !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #555 !important; outline: none !important;
}

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    gap: 0 !important;
    border-bottom: 1px solid #1e1e1e !important;
}
.stTabs [data-baseweb="tab"] {
    color: #888 !important; font-size: 0.85rem !important;
    font-weight: 500 !important; background: transparent !important;
    padding: 0.7rem 1.4rem !important; border: none !important;
    border-radius: 0 !important;
    transition: color 0.15s !important;
}
.stTabs [data-baseweb="tab"]:hover      { color: #ccc !important; }
.stTabs [aria-selected="true"]          {
    color: #fff !important;
    border-bottom: 2px solid #e50914 !important;
}
.stTabs [data-baseweb="tab-panel"]      { padding: 0 !important; }

/* ---------- Empty state ---------- */
.empty-state {
    text-align: center; padding: 5rem 2rem; color: #555;
}
.empty-state .es-title { font-size: 1.1rem; font-weight: 600; color: #777;
                          margin-bottom: 0.4rem; }
.empty-state .es-sub   { font-size: 0.85rem; }

/* ---------- Taste meter ---------- */
.taste-bar {
    background: #1a1a1a; border-left: 3px solid #e50914;
    border-radius: 0 6px 6px 0; padding: 0.9rem 1.2rem;
    margin: 0.5rem 0 0.2rem; display: flex; align-items: center; gap: 1.2rem;
}
.taste-bar .tb-label { font-size: 0.82rem; color: #aaa; }
.taste-bar .tb-value { font-size: 0.78rem; color: #e50914; margin-top: 3px; }

/* ---------- Rating slider ---------- */
div[data-testid="stSlider"] > div > div > div {
    background: #e50914 !important;
}

/* ---------- Number input ---------- */
[data-testid="stNumberInput"] input {
    background: #1c1c1c !important; color: #e8e8e8 !important;
    border: 1px solid #2e2e2e !important; border-radius: 4px !important;
}

/* ---------- Info box (Rate tab) ---------- */
.info-box {
    background: #161616; border: 1px solid #252525; border-radius: 6px;
    padding: 1.4rem 1.5rem; font-size: 0.83rem; color: #888; line-height: 1.8;
}
.info-box b { color: #ccc; }
.info-box .red { color: #e50914; }

/* ---------- Divider ---------- */
hr { border-color: #1e1e1e !important; margin: 0.3rem 0 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────── API helpers ──────────────────────
@st.cache_data(ttl=120)
def get_recommendations(user_id: int, n: int = 21):
    try:
        r = requests.get(f"{API_URL}/recommend/{user_id}", params={"n": n}, timeout=6)
        return r.json() if r.ok else []
    except Exception:
        return []

@st.cache_data(ttl=300)
def get_popular(n: int = 21):
    try:
        r = requests.get(f"{API_URL}/popular", params={"n": n}, timeout=5)
        return r.json() if r.ok else []
    except Exception:
        return []

@st.cache_data(ttl=300)
def get_by_genre(genre: str, n: int = 14):
    try:
        r = requests.get(f"{API_URL}/movies/search", params={"q": genre, "limit": n}, timeout=5)
        data = r.json() if r.ok else []
        # Normalize score field name for consistency
        for m in data:
            if "avg_rating" in m and "score" not in m:
                m["score"] = m["avg_rating"]
        return data
    except Exception:
        return []

@st.cache_data(ttl=45)
def get_trending(n: int = 14):
    try:
        r = requests.get(f"{API_URL}/trending", params={"n": n}, timeout=5)
        return r.json() if r.ok else []
    except Exception:
        return []

def get_from_likes(liked: dict, n: int = 14):
    if not liked:
        return []
    try:
        payload = {
            "likes": [
                {"movie_id": mid,
                 "weight": min(max(float(info.get("score") or 1.0), 0.1), 5.0)}
                for mid, info in liked.items()
            ],
            "n": n,
        }
        r = requests.post(f"{API_URL}/recommend/from-likes", json=payload, timeout=8)
        return r.json() if r.ok else []
    except Exception:
        return []

def post_rating(user_id: int, movie_id: int, rating: float) -> bool:
    try:
        r = requests.post(
            f"{API_URL}/rate",
            json={"user_id": user_id, "movie_id": movie_id, "rating": rating},
            timeout=5,
        )
        return r.ok
    except Exception:
        return False

def get_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.json() if r.ok else {}
    except Exception:
        return {}


# ─────────────────────────── Card renderer ────────────────────
def render_card(col, movie: dict, key_prefix: str):
    mid    = int(movie.get("movie_id") or 0)
    title  = str(movie.get("title") or f"Movie {mid}")
    genres = str(movie.get("genres") or "")
    poster = movie.get("poster_url")
    is_liked = mid in st.session_state.liked_movies

    liked_cls  = "liked-outline" if is_liked else ""
    liked_dot  = '<div class="liked-dot"></div>' if is_liked else ""

    with col:
        if poster:
            st.markdown(f"""
            <div class="card-outer">
              {liked_dot}
              <div class="card-inner {liked_cls}">
                <img src="{poster}" alt="{title}" loading="lazy">
                <div class="card-hover-overlay">
                  <div class="card-title-ov">{title[:36]}</div>
                  <div class="card-genre-ov">{genres[:28]}</div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            short = title[:30] + ("…" if len(title) > 30 else "")
            st.markdown(f"""
            <div class="card-outer">
              {liked_dot}
              <div class="card-inner {liked_cls}">
                <div class="card-no-poster">
                  <div class="np-title">{short}</div>
                </div>
                <div class="card-hover-overlay">
                  <div class="card-title-ov">{title[:36]}</div>
                  <div class="card-genre-ov">{genres[:28]}</div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

        btn_label = "Remove" if is_liked else "Add to My List"
        if st.button(btn_label, key=f"{key_prefix}_{mid}", use_container_width=True):
            if is_liked:
                st.session_state.liked_movies.pop(mid, None)
            else:
                score = float(movie.get("score") or movie.get("avg_rating") or 3.5)
                st.session_state.liked_movies[mid] = {
                    "title": title, "genres": genres,
                    "poster_url": poster, "score": score,
                }
            st.rerun()


def render_row(heading: str, movies: list, key_prefix: str,
               n_cols: int = 7, accent: bool = False, badge: str = ""):
    if not movies:
        return
    badge_html = f'<span class="taste-tag">{badge}</span>' if badge else ""
    cls = "section-title accent" if accent else "section-title"
    st.markdown(f'<div class="{cls}">{heading}{badge_html}</div>', unsafe_allow_html=True)
    cols = st.columns(n_cols)
    for i, movie in enumerate(movies[:n_cols]):
        render_card(cols[i], movie, key_prefix=f"{key_prefix}_{i}")


# ─────────────────────────── Top navigation ───────────────────
health = get_health()
liked  = st.session_state.liked_movies
n_liked = len(liked)

nav_l, nav_mid, nav_r = st.columns([2, 5, 3])
with nav_l:
    st.markdown('<div class="nav-logo">CINEREC</div>', unsafe_allow_html=True)

with nav_mid:
    search_query = st.text_input(
        "search", placeholder="Search movies, genres, directors…",
        label_visibility="collapsed"
    )

with nav_r:
    uid_col, status_col = st.columns([3, 2])
    with uid_col:
        new_uid = st.number_input(
            "User ID", min_value=1, max_value=162541,
            value=st.session_state.user_id, step=1,
            label_visibility="visible"
        )
        if new_uid != st.session_state.user_id:
            st.session_state.user_id = int(new_uid)
            get_recommendations.clear()
            st.rerun()
    with status_col:
        redis_ok = health.get("redis", False)
        movies_n = health.get("movies_loaded", 0)
        st.markdown(f"""
        <div style="font-size:0.7rem;color:#555;margin-top:1.9rem;line-height:1.8">
          {"Connected" if health.get("status")=="ok" else "Offline"}<br>
          {movies_n:,} movies
        </div>""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

user_id = st.session_state.user_id

# ══════════════════════════════════════════════════════════════
# SEARCH MODE
# ══════════════════════════════════════════════════════════════
if search_query and len(search_query.strip()) >= 2:
    results = get_by_genre(search_query.strip(), n=28)
    count = len(results)
    st.markdown(
        f'<div class="section-title">Results for "{search_query}"'
        f'<span class="count">{count} found</span></div>',
        unsafe_allow_html=True,
    )
    if results:
        n_cols = 7
        for row_start in range(0, min(count, 28), n_cols):
            row = results[row_start:row_start + n_cols]
            cols = st.columns(n_cols)
            for i, movie in enumerate(row):
                render_card(cols[i], movie, key_prefix=f"s_{row_start}_{i}")
    else:
        st.markdown('<div class="empty-state"><div class="es-title">No results</div>'
                    '<div class="es-sub">Try a different title or genre</div></div>',
                    unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# BROWSE MODE
# ══════════════════════════════════════════════════════════════
else:
    tab_home, tab_list, tab_trending, tab_rate = st.tabs([
        "Home",
        f"My List  {('· ' + str(n_liked)) if n_liked else ''}",
        "Trending",
        "Rate",
    ])

    # ── HOME ──────────────────────────────────────────────────
    with tab_home:
        # Hero
        popular = get_popular(n=21)
        hero = next((m for m in popular if m.get("poster_url")), popular[0] if popular else None)

        if hero:
            ov = hero.get("overview") or "A critically acclaimed film you will love."
            st.markdown(f"""
            <div class="hero-wrap">
              <img class="hero-img" src="{hero['poster_url']}" alt="">
              <div class="hero-grad"></div>
              <div class="hero-body">
                <div class="hero-title">{hero.get('title','')}</div>
                <div class="hero-meta">{hero.get('genres','')[:55]}</div>
                <div class="hero-desc">{ov[:220]}</div>
                <div class="hero-actions">
                  <button class="btn-play">Play</button>
                  <button class="btn-more">More Info</button>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

        # Taste model row — only when user has likes
        if liked:
            quality_levels = [
                (1, "Add more titles to strengthen your taste model"),
                (3, "Taste model is building — each title refines the ALS vector"),
                (6, "Good taste profile — recommendations are well calibrated"),
                (99, "Strong taste model — ALS solver is well conditioned"),
            ]
            quality_msg = next(msg for threshold, msg in quality_levels if n_liked <= threshold)

            titles_preview = ", ".join(
                info["title"][:18] for info in list(liked.values())[:3]
            )
            if n_liked > 3:
                titles_preview += f" + {n_liked - 3} more"

            st.markdown(f"""
            <div class="taste-bar">
              <div>
                <div class="tb-label">
                  Taste model trained on {n_liked} title{"s" if n_liked > 1 else ""}:
                  <span style="color:#ccc">{titles_preview}</span>
                </div>
                <div class="tb-value">{quality_msg}</div>
              </div>
            </div>""", unsafe_allow_html=True)

            with st.spinner("Computing recommendations…"):
                like_recs = get_from_likes(liked, n=14)
            render_row("Recommended For You", like_recs, "lk",
                       accent=True, badge="ALS Taste Model")

        # Personalised row
        with st.spinner("Loading your picks…"):
            user_recs = get_recommendations(user_id, n=14)
        render_row(f"Top Picks for You", user_recs, "ur")

        # Genre rows
        GENRES = [
            ("Drama",           "Drama"),
            ("Action",          "Action"),
            ("Comedy",          "Comedy"),
            ("Science Fiction", "Sci-Fi"),
            ("Thriller",        "Thriller"),
            ("Romance",         "Romance"),
            ("Horror",          "Horror"),
            ("Crime",           "Crime"),
            ("Animation",       "Animation"),
            ("Documentary",     "Documentary"),
            ("War",             "War"),
        ]
        for row_title, genre_key in GENRES:
            movies = get_by_genre(genre_key, n=7)
            render_row(row_title, movies, f"g_{genre_key}")

        render_row("Most Popular", popular, "pop")

    # ── MY LIST ───────────────────────────────────────────────
    with tab_list:
        if not liked:
            st.markdown("""
            <div class="empty-state">
              <div class="es-title">Your list is empty</div>
              <div class="es-sub">Add titles from the Home tab to train your taste model</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="section-title">'
                f'My List<span class="count">{n_liked} title{"s" if n_liked>1 else ""}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            n_cols = 7
            liked_list = [{"movie_id": mid, **info} for mid, info in liked.items()]
            for row_start in range(0, len(liked_list), n_cols):
                batch = liked_list[row_start:row_start + n_cols]
                cols  = st.columns(n_cols)
                for i, movie in enumerate(batch):
                    render_card(cols[i], movie, key_prefix=f"ml2_{row_start}_{i}")

            st.markdown("<br>", unsafe_allow_html=True)
            with st.spinner("Computing recommendations from your list…"):
                taste_recs = get_from_likes(liked, n=14)
            render_row("Because You Added These", taste_recs, "bl",
                       accent=True, badge="ALS Taste Model")

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Clear My List", use_container_width=False):
                st.session_state.liked_movies.clear()
                st.rerun()

    # ── TRENDING ──────────────────────────────────────────────
    with tab_trending:
        trending_data = get_trending(n=14)
        has_live = any(m.get("event_count") for m in trending_data)
        if trending_data:
            badge = "Live" if has_live else ""
            render_row("Trending Now", trending_data[:7], "tr1", badge=badge)
            if len(trending_data) > 7:
                render_row("Also Popular", trending_data[7:14], "tr2")
        else:
            st.markdown("""
            <div class="empty-state">
              <div class="es-title">No live data</div>
              <div class="es-sub">Run <code>make stream-producer</code> to stream user events
              through Kafka into the trending feed</div>
            </div>""", unsafe_allow_html=True)

    # ── RATE ──────────────────────────────────────────────────
    with tab_rate:
        st.markdown('<div class="section-title">Rate a Movie</div>', unsafe_allow_html=True)

        r_col1, r_col2 = st.columns([1, 1])
        with r_col1:
            rate_id  = st.number_input("Movie ID", min_value=1, value=1, step=1)
            rate_val = st.slider("Rating", 0.5, 5.0, 4.0, 0.5)
            if st.button("Submit Rating", type="primary", use_container_width=True):
                ok = post_rating(user_id, int(rate_id), rate_val)
                if ok:
                    st.success(f"Submitted: {rate_val} stars for movie {rate_id}")
                    get_recommendations.clear()
                else:
                    st.error("Submission failed — check API is running")

        with r_col2:
            st.markdown("""
            <div class="info-box">
              <b>How the recommendation pipeline works</b><br><br>
              1. Your rating hits <b>POST /rate</b> on the FastAPI server<br>
              2. The API publishes it to the <b class="red">Kafka</b> topic
                 <code>user-events</code><br>
              3. <b>Spark Structured Streaming</b> consumes events in real time,
                 aggregates trending windows, and writes results to Redis<br>
              4. The Trending tab reflects live engagement within seconds<br><br>
              <b>The model</b><br><br>
              <span class="red">Spark MLlib ALS</span> — trained on
              <b>25 million ratings</b> from MovieLens 25M and 7.4 million
              Amazon review ratings. The algorithm learns 100-dimensional
              latent vectors for every user and movie. The
              <b>Add to My List</b> feature solves for your personal user
              vector using the ALS update equation in real time — no
              retraining required.
            </div>""", unsafe_allow_html=True)
