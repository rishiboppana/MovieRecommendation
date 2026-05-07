"""
CineRec — Netflix-style Movie Recommendation UI
Model: Spark MLlib ALS trained on 25M ratings (MovieLens 25M dataset)
"""
import os
import html
import sqlite3
import requests
import pandas as pd
import pyarrow.parquet as pq
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="CineRec",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Session state ──
if "all_liked_movies" not in st.session_state:
    st.session_state.all_liked_movies = {}  # {user_id: {movie_id: info}}
if "user_id" not in st.session_state:
    st.session_state.user_id = 1
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "home"
if "selected_movie_id" not in st.session_state:
    st.session_state.selected_movie_id = None
if "rate_reset" not in st.session_state:
    st.session_state.rate_reset = 0
if "search_reset" not in st.session_state:
    st.session_state.search_reset = 0
if "rate_selected_movie" not in st.session_state:
    st.session_state.rate_selected_movie = None

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

/* ---------- Detail panel ---------- */
.detail-panel {
    background: #141414;
    border: 1px solid #1e1e1e;
    border-radius: 10px;
    padding: 1.5rem;
    margin: 0.6rem 0 1rem;
}
.detail-title {
    font-size: 1.9rem; font-weight: 700; color: #fff;
    margin: 0.3rem 0 0.5rem; line-height: 1.15;
}
.detail-meta   { font-size: 0.8rem; color: #666; letter-spacing: 0.3px; }
.detail-rating { font-size: 0.95rem; color: #e50914; margin-bottom: 0.9rem; }
.detail-overview {
    font-size: 0.9rem; color: #bbb; line-height: 1.75; max-width: 72ch;
}
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
        # Use uniform weight=1.0 for each like so the ALS solver aggregates
        # user taste across titles rather than amplifying movies with higher
        # precomputed 'score' values stored in the UI state.
        payload = {
            "likes": [
                {"movie_id": mid, "weight": 1.0}
                for mid in liked.keys()
            ],
            "n": n,
        }
        r = requests.post(f"{API_URL}/recommend/from-likes", json=payload, timeout=8)
        return r.json() if r.ok else []
    except Exception:
        return []

def post_rating(
    user_id: int, movie_id: int, rating: float,
    review: str = "", username: str = "",
) -> bool:
    try:
        r = requests.post(
            f"{API_URL}/rate",
            json={
                "user_id": user_id,
                "movie_id": movie_id,
                "rating": rating,
                "review": review.strip() or None,
                "username": username.strip() or None,
            },
            timeout=5,
        )
        return r.ok
    except Exception:
        return False


@st.cache_data(ttl=30, show_spinner=False)
def get_reviews(movie_id: int, limit: int = 15):
    try:
        r = requests.get(
            f"{API_URL}/reviews/{movie_id}", params={"limit": limit}, timeout=5
        )
        return r.json() if r.ok else []
    except Exception:
        return []

def get_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.json() if r.ok else {}
    except Exception:
        return {}

@st.cache_data(ttl=3600, show_spinner=False)
def get_movie_rating_dist(movie_id: int):
    """Load star-rating distribution for a specific movie from the gold parquet."""
    try:
        path = _DATA_ROOT / "data" / "processed" / "gold" / "ratings"
        table = pq.read_table(str(path), filters=[("movie_id", "=", movie_id)])
        df = table.to_pandas()
        dist = df["rating"].value_counts().sort_index().reset_index()
        dist.columns = ["rating", "count"]
        dist["pct"] = (dist["count"] / dist["count"].sum() * 100).round(1)
        return dist
    except Exception:
        return None


@st.cache_data(ttl=600, show_spinner=False)
def get_movie_detail(movie_id: int):
    try:
        r = requests.get(f"{API_URL}/movies/{movie_id}", timeout=5)
        return r.json() if r.ok else {}
    except Exception:
        return {}

@st.cache_data(ttl=600, show_spinner=False)
def get_similar(movie_id: int, n: int = 7):
    try:
        r = requests.get(f"{API_URL}/similar/{movie_id}", params={"n": n}, timeout=5)
        return r.json() if r.ok else []
    except Exception:
        return []


# ─────────────────────────── Analytics data ───────────────────
_DATA_ROOT = Path(__file__).parent.parent

_PLOT_BASE = dict(
    paper_bgcolor="#111111",
    plot_bgcolor="#111111",
    font=dict(family="Inter, -apple-system, sans-serif", color="#999", size=11),
    margin=dict(l=4, r=4, t=36, b=4),
    xaxis=dict(gridcolor="#1e1e1e", zerolinecolor="#1e1e1e"),
    yaxis=dict(gridcolor="#1e1e1e", zerolinecolor="#1e1e1e"),
    colorway=["#e50914", "#c5000f", "#888", "#555", "#333"],
)


@st.cache_data(ttl=3600, show_spinner=False)
def load_analytics():
    out = {}

    mp = _DATA_ROOT / "data" / "processed" / "movie_features"
    if mp.exists():
        out["movies"] = pq.read_table(str(mp)).to_pandas()

    up = _DATA_ROOT / "data" / "processed" / "user_features"
    if up.exists():
        out["users"] = pq.read_table(str(up)).to_pandas()

    gp = _DATA_ROOT / "data" / "processed" / "gold" / "ratings"
    if gp.exists():
        raw = pq.read_table(str(gp), columns=["rating"]).to_pandas()
        dist = raw["rating"].value_counts().sort_index().reset_index()
        dist.columns = ["rating", "count"]
        out["rating_dist"] = dist
        del raw

    db = _DATA_ROOT / "mlflow" / "mlflow.db"
    if db.exists():
        try:
            conn = sqlite3.connect(str(db))
            # Pick metrics from the final production run only (not HPO trials)
            mdf = pd.read_sql("""
                SELECT m.key, m.value
                FROM metrics m
                JOIN runs r ON m.run_uuid = r.run_uuid
                WHERE r.status = 'FINISHED'
                  AND r.name NOT LIKE 'ray-tune-%'
                ORDER BY r.start_time DESC
            """, conn)
            conn.close()
            out["metrics"] = dict(zip(mdf["key"], mdf["value"]))
        except Exception:
            out["metrics"] = {}

    return out


def _genre_stats(movies: pd.DataFrame) -> pd.DataFrame:
    exploded = (
        movies[["movie_id", "genres", "rating_count", "avg_rating"]]
        .assign(genre=movies["genres"].str.split("|"))
        .explode("genre")
    )
    exploded = exploded[exploded["genre"].notna() & (exploded["genre"] != "") & (exploded["genre"] != "(no genres listed)")]
    return (
        exploded.groupby("genre")
        .agg(movie_count=("movie_id", "count"),
             total_ratings=("rating_count", "sum"),
             avg_rating=("avg_rating", "mean"))
        .sort_values("total_ratings", ascending=False)
        .reset_index()
    )


# ─────────────────────────── Card renderer ────────────────────
def render_card(col, movie: dict, key_prefix: str):
    mid    = int(movie.get("movie_id") or 0)
    title  = html.escape(str(movie.get("title") or f"Movie {mid}"))
    genres = html.escape(str(movie.get("genres") or ""))
    poster = movie.get("poster_url")
    _uid   = st.session_state.user_id
    _liked = st.session_state.all_liked_movies.setdefault(_uid, {})
    is_liked = mid in _liked

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

        d_col, l_col = st.columns(2)
        with d_col:
            if st.button("Details", key=f"dt_{key_prefix}_{mid}", use_container_width=True):
                st.session_state.selected_movie_id = mid
                st.rerun()
        with l_col:
            btn_label = "Remove" if is_liked else "Add to List"
            if st.button(btn_label, key=f"{key_prefix}_{mid}", use_container_width=True):
                if is_liked:
                    _liked.pop(mid, None)
                else:
                    score = float(movie.get("score") or movie.get("avg_rating") or 3.5)
                    _liked[mid] = {
                        "title": title, "genres": genres,
                        "poster_url": poster, "score": score,
                    }


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


def render_detail_panel():
    mid = st.session_state.selected_movie_id
    if mid is None:
        return

    with st.spinner("Loading details…"):
        meta    = get_movie_detail(mid)
        similar = get_similar(mid, n=7)

    title          = str(meta.get("title") or f"Movie {mid}")
    poster         = meta.get("poster_url")
    overview       = str(meta.get("overview") or "No overview available.")
    genres         = str(meta.get("genres") or "")
    year           = meta.get("year")
    avg_rating     = meta.get("avg_rating")
    rating_count   = meta.get("rating_count")
    genome_themes  = meta.get("genome_themes")

    meta_parts = [str(int(year)) if year else "", genres]
    meta_str   = " · ".join(p for p in meta_parts if p)

    rating_str = ""
    if avg_rating is not None:
        rating_str = f"{float(avg_rating):.2f} ★"
        if rating_count:
            rating_str += f"  ·  {int(rating_count):,} ratings"

    st.markdown('<div class="detail-panel">', unsafe_allow_html=True)

    # Close button in its own column so it doesn't stretch full-width
    hdr_col, close_col = st.columns([10, 1])
    with close_col:
        if st.button("✕ Close", key="detail_close", use_container_width=True):
            st.session_state.selected_movie_id = None
            st.rerun()

    img_col, info_col = st.columns([1, 3])

    with img_col:
        if poster:
            st.image(poster, use_column_width=True)
        else:
            st.markdown(
                '<div style="width:100%;aspect-ratio:2/3;background:#1c1c1c;'
                'border-radius:6px;display:flex;align-items:center;'
                'justify-content:center;color:#555;font-size:0.8rem;">No poster</div>',
                unsafe_allow_html=True,
            )

    with info_col:
        genome_html = ""
        if genome_themes:
            tags = [t.strip() for t in genome_themes.split("|") if t.strip()]
            tag_pills = "".join(
                f'<span style="display:inline-block;background:#1c1c1c;border:1px solid #333;'
                f'border-radius:12px;padding:2px 10px;margin:2px 3px 2px 0;font-size:0.75rem;'
                f'color:#aaa;">{t}</span>'
                for t in tags
            )
            genome_html = f'<div style="margin-top:8px;">{tag_pills}</div>'

        st.markdown(
            f'<div class="detail-meta">{meta_str}</div>'
            f'<div class="detail-title">{title}</div>'
            f'<div class="detail-rating">{rating_str}</div>'
            f'<div class="detail-overview">{overview}</div>'
            f'{genome_html}',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        # Add to My List button inside the detail panel
        _uid_det  = st.session_state.user_id
        _liked_det = st.session_state.all_liked_movies.setdefault(_uid_det, {})
        is_liked  = mid in _liked_det
        btn_lbl   = "▼ Remove from My List" if is_liked else "+ Add to My List"
        if st.button(btn_lbl, key=f"det_list_{mid}"):
            if is_liked:
                _liked_det.pop(mid, None)
            else:
                score = float(meta.get("avg_rating") or 3.5)
                _liked_det[mid] = {
                    "title": title, "genres": genres,
                    "poster_url": poster, "score": score,
                }

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Community Reviews (full-width, 2-col grid) ──
    st.markdown(
        '<div class="section-title" style="margin-top:1.2rem;">Community Reviews</div>',
        unsafe_allow_html=True,
    )
    det_reviews = get_reviews(mid, limit=10)
    if det_reviews:
        rv_left, rv_right = st.columns(2)
        for idx, rv in enumerate(det_reviews):
            stars        = "★" * round(rv["rating"]) + "☆" * (5 - round(rv["rating"]))
            date_str     = rv["created_at"][:10] if rv.get("created_at") else ""
            review_body  = rv.get("review") or ""
            display_name = rv.get("username") or f"User {rv['user_id']}"
            body_html = (
                f'<div style="font-size:0.8rem;color:#aaa;line-height:1.6;margin-top:0.25rem;">'
                f'{review_body}</div>'
                if review_body else
                '<div style="font-size:0.72rem;color:#555;font-style:italic;">No written review</div>'
            )
            card_html = (
                f'<div style="background:#161616;border:1px solid #222;border-radius:6px;'
                f'padding:0.7rem 0.9rem;margin-bottom:0.45rem;">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                f'<span style="color:#e8a010;font-size:0.82rem;letter-spacing:1px;">{stars}</span>'
                f'<span style="font-size:0.68rem;color:#555;">{display_name} · {date_str}</span>'
                f'</div>{body_html}</div>'
            )
            target = rv_left if idx % 2 == 0 else rv_right
            target.markdown(card_html, unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="font-size:0.8rem;color:#555;font-style:italic;margin-bottom:0.8rem;">'
            'No reviews yet.</div>',
            unsafe_allow_html=True,
        )

    # ── Similar Movies (full-width, outside any column context) ──
    if similar:
        render_row("Similar Movies", similar, "sim_det", n_cols=7)

    st.markdown("<hr>", unsafe_allow_html=True)


def _render_rate_thumb(col, movie: dict, key_prefix: str):
    """Compact selectable thumbnail for the Rate tab."""
    mid = int(movie.get("movie_id") or 0)
    title = str(movie.get("title") or f"Movie {mid}")
    poster = movie.get("poster_url")
    is_selected = (
        st.session_state.rate_selected_movie is not None
        and int(st.session_state.rate_selected_movie.get("movie_id", -1)) == mid
    )
    border = "border:2px solid #e50914;" if is_selected else "border:2px solid transparent;"
    short = title[:20] + ("…" if len(title) > 20 else "")
    with col:
        if poster:
            st.markdown(
                f'<div style="aspect-ratio:2/3;overflow:hidden;border-radius:4px;{border}">'
                f'<img src="{poster}" style="width:100%;height:100%;object-fit:cover;" /></div>'
                f'<div style="font-size:0.6rem;color:#aaa;text-align:center;margin-top:2px;'
                f'overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{short}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div style="aspect-ratio:2/3;background:#1c1c1c;border-radius:4px;{border}'
                f'display:flex;align-items:center;justify-content:center;'
                f'font-size:0.6rem;color:#888;text-align:center;padding:0.3rem;">'
                f'{short}</div>',
                unsafe_allow_html=True,
            )
        btn_lbl = "✓" if is_selected else "Select"
        if st.button(btn_lbl, key=f"{key_prefix}_{mid}", use_container_width=True):
            st.session_state.rate_selected_movie = movie
            st.rerun()


# ─────────────────────────── Top navigation ───────────────────
health = get_health()
liked  = st.session_state.all_liked_movies.setdefault(st.session_state.user_id, {})
n_liked = len(liked)

nav_l, nav_mid, nav_r = st.columns([2, 5, 3])
with nav_l:
    st.markdown('<div class="nav-logo">CINEREC</div>', unsafe_allow_html=True)

with nav_mid:
    search_query = st.text_input(
        "search", placeholder="Search movie titles or genres…",
        label_visibility="collapsed",
        key=f"main_search_{st.session_state.search_reset}",
    )

_DEMO_PROFILES = {
    "Demo User A": 42,
    "Demo User B": 500,
    "Demo User C": 7777,
    "Demo User D": 50000,
    "Demo User E": 120000,
    "Custom…": None,
}

with nav_r:
    profile_col, status_col = st.columns([3, 2])
    with profile_col:
        selected_profile = st.selectbox(
            "Profile",
            options=list(_DEMO_PROFILES.keys()),
            key="profile_select",
            label_visibility="visible",
        )
        preset_uid = _DEMO_PROFILES[selected_profile]
        if preset_uid is not None:
            new_uid = preset_uid
        else:
            new_uid = st.number_input(
                "User ID", min_value=1, max_value=162541,
                value=st.session_state.user_id, step=1,
                label_visibility="collapsed",
            )
        if int(new_uid) != st.session_state.user_id:
            st.session_state.user_id = int(new_uid)
            get_recommendations.clear()
            st.rerun()
    with status_col:
        movies_n = health.get("movies_loaded", 0)
        st.markdown(f"""
        <div style="font-size:0.7rem;color:#555;margin-top:1.9rem;line-height:1.8">
          {"Connected" if health.get("status")=="ok" else "Offline"}<br>
          {movies_n:,} movies
        </div>""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

user_id = st.session_state.user_id

# ── Detail panel (persists across search and browse modes) ────
if st.session_state.selected_movie_id:
    render_detail_panel()

# ══════════════════════════════════════════════════════════════
# SEARCH MODE
# ══════════════════════════════════════════════════════════════
if search_query and len(search_query.strip()) >= 2:
    _back_col, _ = st.columns([1, 9])
    with _back_col:
        if st.button("← Back", key="search_back", use_container_width=True):
            st.session_state.search_reset += 1
            st.rerun()
    results = get_by_genre(search_query.strip(), n=28)
    count = len(results)
    st.markdown(
        f'<div class="section-title">Results for "{search_query}"'
        f'<span class="count">{count} found</span></div>',
        unsafe_allow_html=True,
    )
    if results:
        # Feature the best match for faster scanning
        top = results[0]
        top_title = str(top.get("title") or "")
        top_genres = str(top.get("genres") or "")
        top_avg = top.get("avg_rating")
        top_cnt = top.get("rating_count")
        top_match = top.get("match_score")
        top_poster = top.get("poster_url")

        st.markdown(
            """
            <div style="display:flex; gap:1.2rem; align-items:flex-start;
                        background:#141414; border:1px solid #222; border-radius:10px;
                        padding:1rem; margin:0.8rem 0 1.2rem;">
              <div style="width:120px; flex:0 0 120px;">
            """,
            unsafe_allow_html=True,
        )
        if top_poster:
            st.markdown(
                f'<img src="{top_poster}" style="width:120px; border-radius:8px; display:block;" />',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="width:120px; height:180px; border-radius:8px;'
                ' background:#1f1f1f; display:flex; align-items:center; justify-content:center;'
                ' color:#666; font-size:0.8rem;">No poster</div>',
                unsafe_allow_html=True,
            )

        # Right side details
        stats = []
        if top_avg is not None:
            stats.append(f"Avg rating: {float(top_avg):.2f}")
        if top_cnt is not None:
            stats.append(f"Ratings: {int(top_cnt):,}")
        if top_match is not None:
            stats.append(f"Match: {float(top_match):.2f}")
        stats_str = " · ".join(stats) if stats else ""

        st.markdown(
            f"""
              </div>
              <div style="flex:1 1 auto; min-width:0;">
                <div style="font-size:0.75rem; color:#777; letter-spacing:0.3px;">Top match</div>
                <div style="font-size:1.35rem; font-weight:700; color:#e50914; margin-top:0.15rem;
                            word-break:break-word;">
                  {top_title}
                </div>
                <div style="font-size:0.9rem; color:#aaa; margin-top:0.25rem;">
                  {top_genres}
                </div>
                <div style="font-size:0.82rem; color:#666; margin-top:0.45rem;">
                  {stats_str}
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

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
    tab_home, tab_list, tab_trending, tab_rate, tab_analytics = st.tabs([
        "Home",
        "My List",
        "Trending",
        "Rate",
        "Analytics",
    ])

    # ── HOME ──────────────────────────────────────────────────
    with tab_home:
        # Hero
        popular = get_popular(n=21)
        _poster_picks = [m for m in popular if m.get("poster_url")]
        _hero_idx = pd.Timestamp.now().dayofyear % max(4, min(7, len(_poster_picks)))
        hero = _poster_picks[_hero_idx] if _poster_picks else (popular[0] if popular else None)

        if hero:
            hero_mid = int(hero.get("movie_id") or 0)
            ov = html.escape(hero.get("overview") or "A critically acclaimed film you will love.")
            hero_title = html.escape(hero.get('title', ''))
            hero_meta = html.escape(hero.get('genres', '')[:55])
            poster = hero.get("poster_url")
            if poster:
                hero_img = f'<img class="hero-img" src="{html.escape(poster)}" alt="">'
            else:
                hero_img = '<div class="hero-img" style="background:#1a1a1a;width:100%;height:100%"></div>'
            st.markdown(f"""
            <div class="hero-wrap">
              {hero_img}
              <div class="hero-grad"></div>
              <div class="hero-body">
                <div class="hero-title">{hero_title}</div>
                <div class="hero-meta">{hero_meta}</div>
                <div class="hero-desc">{ov[:220]}</div>
              </div>
            </div>""", unsafe_allow_html=True)

            # Real Streamlit buttons overlaid at bottom-left of hero via CSS
            st.markdown("""<style>
            div[data-testid="stHorizontalBlock"].hero-btn-row {
                margin-top:-4.6rem !important;
                padding-left:4.2% !important;
                margin-bottom:3.8rem !important;
                position:relative; z-index:10;
            }
            div[data-testid="stHorizontalBlock"].hero-btn-row button {
                border-radius:4px !important; font-weight:600 !important;
                font-size:0.88rem !important; padding:0.55rem 0 !important;
            }
            </style>""", unsafe_allow_html=True)
            st.markdown('<div class="hero-btn-row">', unsafe_allow_html=True)
            _pc, _mc, _ = st.columns([1, 1, 8])
            with _pc:
                st.button("▶  Play", key="hero_play", use_container_width=True)
            with _mc:
                if st.button("ℹ  More Info", key="hero_more_info", use_container_width=True):
                    st.session_state.selected_movie_id = hero_mid
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

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
            ("Science Fiction", "Science Fiction"),
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
                st.session_state.all_liked_movies.get(user_id, {}).clear()
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

        rate_search = st.text_input(
            "Search for a movie",
            placeholder="e.g., Inception, The Matrix, Toy Story...",
            key=f"rate_search_{st.session_state.rate_reset}",
        )

        if rate_search and len(rate_search) >= 2:
            with st.spinner("Searching…"):
                try:
                    _rate_results = requests.get(
                        f"{API_URL}/movies/search",
                        params={"q": rate_search, "limit": 18},
                        timeout=5,
                    ).json()
                except Exception as _e:
                    _rate_results = []
                    st.error(f"Search failed: {_e}")

            if _rate_results:
                st.markdown(
                    '<div style="font-size:0.78rem;color:#666;margin:0.5rem 0 0.3rem;">'
                    'Click <b style="color:#ccc">Select</b> on a movie below</div>',
                    unsafe_allow_html=True,
                )
                _n_rt = 7
                for _rrow in range(0, len(_rate_results), _n_rt):
                    _rt_batch = _rate_results[_rrow:_rrow + _n_rt]
                    _rt_cols = st.columns(_n_rt)
                    for _ri, _rm in enumerate(_rt_batch):
                        _render_rate_thumb(_rt_cols[_ri], _rm, f"rt_{_rrow}_{_ri}")
            else:
                st.info("No movies found. Try a different search.")
        elif rate_search and len(rate_search) < 2:
            st.info("Type at least 2 characters to search")

        # ── Selected movie details + rating ──
        _sel = st.session_state.rate_selected_movie
        if _sel:
            st.markdown("<hr>", unsafe_allow_html=True)
            _det_col, _rev_col = st.columns([1, 1])

            with _det_col:
                rate_id = _sel.get("movie_id")
                avg_r   = float(_sel.get("avg_rating") or 0)
                n_r     = int(_sel.get("rating_count") or 0)
                filled  = "★" * round(avg_r) + "☆" * (5 - round(avg_r))
                st.markdown(f"""
                <div style="background:#1a1a1a;padding:1rem;border-radius:6px;
                            margin-top:0.5rem;border-left:3px solid #e50914;">
                  <div style="font-size:0.78rem;color:#777;">Selected</div>
                  <div style="font-size:1.15rem;font-weight:700;color:#e50914;margin:0.2rem 0;">
                    {_sel['title']}
                  </div>
                  <div style="font-size:0.9rem;color:#e8a010;letter-spacing:2px;">{filled}</div>
                  <div style="font-size:0.78rem;color:#666;margin-top:0.3rem;">
                    Avg: {avg_r:.2f} ★ &nbsp;·&nbsp; {n_r:,} ratings &nbsp;·&nbsp; ID: {rate_id}
                  </div>
                </div>""", unsafe_allow_html=True)

                with st.spinner("Loading ratings…"):
                    rdist = get_movie_rating_dist(int(rate_id))
                if rdist is not None and not rdist.empty:
                    st.markdown(
                        '<div style="font-size:0.78rem;color:#666;margin:0.8rem 0 0.2rem;">'
                        'How others rated this movie</div>',
                        unsafe_allow_html=True,
                    )
                    fig_rd = px.bar(
                        rdist, x="rating", y="count",
                        text=rdist["pct"].astype(str) + "%",
                        labels={"rating": "Stars", "count": "Ratings"},
                        color_discrete_sequence=["#e50914"],
                        height=160,
                    )
                    fig_rd.update_layout(
                        **{k: v for k, v in _PLOT_BASE.items() if k not in ("margin",)},
                        margin=dict(l=0, r=0, t=4, b=0),
                        showlegend=False,
                        title=None,
                    )
                    fig_rd.update_traces(
                        marker_line_width=0,
                        textposition="outside",
                        textfont=dict(color="#666", size=9),
                    )
                    st.plotly_chart(fig_rd, use_container_width=True)

                st.markdown("<br>", unsafe_allow_html=True)
                rate_val = st.slider(
                    "Your Rating", 0.5, 5.0, 3.0, 0.5,
                    key=f"rate_slider_{st.session_state.rate_reset}",
                )
                review_text = st.text_area(
                    "Write a review (optional)",
                    placeholder="What did you think? Any standout scenes, performances, or reasons you'd recommend it?",
                    height=120,
                    key=f"rate_review_{st.session_state.rate_reset}",
                )
                if st.button("Submit Rating", type="primary", use_container_width=True):
                    ok = post_rating(user_id, int(rate_id), rate_val, review_text,
                                     username=selected_profile)
                    if ok:
                        rev_note = " · Review saved ✍" if review_text.strip() else ""
                        st.success(f"✓ Submitted {rate_val}★ for {_sel['title']}{rev_note}")
                        get_recommendations.clear()
                        get_reviews.clear()
                        st.session_state.rate_selected_movie = None
                        st.session_state.rate_reset += 1
                        st.rerun()
                    else:
                        st.error("Submission failed — check API is running")

            with _rev_col:
                st.markdown(
                    '<div style="font-size:0.9rem;font-weight:600;color:#ccc;'
                    'margin-top:0.5rem;margin-bottom:0.6rem;">Community Reviews</div>',
                    unsafe_allow_html=True,
                )
                reviews = get_reviews(int(rate_id))
                if reviews:
                    for rv in reviews:
                        stars = "★" * round(rv["rating"]) + "☆" * (5 - round(rv["rating"]))
                        date_str = rv["created_at"][:10] if rv.get("created_at") else ""
                        review_body = rv.get("review") or ""
                        display_name = rv.get("username") or f"User {rv['user_id']}"
                        body_html = (
                            f'<div style="font-size:0.82rem;color:#aaa;line-height:1.6;">{review_body}</div>'
                            if review_body else
                            '<div style="font-size:0.75rem;color:#555;font-style:italic;">No written review</div>'
                        )
                        st.markdown(
                            f'<div style="background:#161616;border:1px solid #222;'
                            f'border-radius:6px;padding:0.8rem 1rem;margin-bottom:0.5rem;">'
                            f'<div style="display:flex;justify-content:space-between;'
                            f'align-items:center;margin-bottom:0.3rem;">'
                            f'<span style="color:#e8a010;letter-spacing:1px;font-size:0.85rem;">{stars}</span>'
                            f'<span style="font-size:0.7rem;color:#555;">{display_name} · {date_str}</span>'
                            f'</div>{body_html}</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.markdown(
                        '<div style="font-size:0.8rem;color:#555;font-style:italic;margin-top:0.5rem;">'
                        'No reviews yet — be the first!</div>',
                        unsafe_allow_html=True,
                    )
        else:
            if not rate_search:
                st.info("👉 Search for a movie above to get started")



    # ── ANALYTICS ─────────────────────────────────────────────
    with tab_analytics:
        with st.spinner("Loading analytics…"):
            adata = load_analytics()

        movies_df  = adata.get("movies")
        users_df   = adata.get("users")
        rating_dist = adata.get("rating_dist")
        al_metrics  = adata.get("metrics", {})

        if movies_df is None:
            st.markdown(
                '<div class="empty-state"><div class="es-title">No data available</div>'
                '<div class="es-sub">Run <code>make etl</code> then <code>make train</code></div></div>',
                unsafe_allow_html=True,
            )
        else:
            # ── Static KPIs (full dataset, no filter yet) ─────────
            total_all  = int(movies_df["rating_count"].sum())
            n_users    = len(users_df) if users_df is not None else 162541

            st.markdown('<div class="section-title">Dataset Overview</div>', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Ratings",   f"{total_all:,}")
            c2.metric("Unique Movies",   f"{len(movies_df):,}")
            c3.metric("Unique Users",    f"{n_users:,}")
            c4.metric("Matrix Sparsity", f"{1.0 - total_all/(n_users*len(movies_df)):.2%}")

            # ── Model Performance KPIs ──
            if al_metrics:
                st.markdown(
                    '<div class="section-title" style="margin-top:1.8rem">'
                    'Model Performance — Spark MLlib ALS</div>',
                    unsafe_allow_html=True,
                )
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("RMSE",       f"{al_metrics.get('rmse', 0):.4f}",
                          help="RMSE on binarised labels (rating ≥ 4.0 = positive)")
                m2.metric("MAE",        f"{al_metrics.get('mae', 0):.4f}")
                m3.metric("MAP@10",     f"{al_metrics.get('map_at_10', 0):.4f}",
                          help="Mean Average Precision at K=10")
                m4.metric("Train size", f"{int(al_metrics.get('train_size', 0)):,}")

            st.markdown("<hr>", unsafe_allow_html=True)

            # ── Live filters (placed here so charts below react visibly) ──
            st.markdown('<div class="section-title">Explore the Data</div>', unsafe_allow_html=True)
            f1, f2, f3 = st.columns(3)
            with f1:
                yr_range = st.slider(
                    "Release year range", 1950, 2019, (1970, 2019),
                    key="ana_year",
                    help="Filters the Movies Per Year and Avg Rating by Year charts",
                )
            with f2:
                min_ratings = st.select_slider(
                    "Min ratings per movie",
                    options=[5, 20, 50, 100, 500, 1000, 5000, 10000],
                    value=50,
                    key="ana_minrat",
                    help="Raise to focus on popular movies only",
                )
            with f3:
                top_n = st.slider(
                    "Top N movies", 5, 50, 20, 5,
                    key="ana_topn",
                    help="How many movies appear in the Top Movies chart",
                )

            # Filtered working copy (raw stays cached)
            filt_df = movies_df[movies_df["rating_count"] >= min_ratings].copy()
            total_ratings = int(filt_df["rating_count"].sum())
            n_movies      = len(filt_df)
            sparsity       = 1.0 - total_ratings / (n_users * n_movies)

            # Quick delta KPIs so the filter effect is immediately obvious
            d1, d2, d3 = st.columns(3)
            d1.metric("Movies matching filter", f"{n_movies:,}",
                      delta=f"{n_movies - len(movies_df):,}", delta_color="off")
            d2.metric("Ratings in filtered set", f"{total_ratings:,}")
            d3.metric("Filtered sparsity",       f"{sparsity:.2%}")

            st.markdown("<hr>", unsafe_allow_html=True)

            # Precompute year_df
            year_df = (
                filt_df[
                    (filt_df["year"] >= yr_range[0]) & (filt_df["year"] <= yr_range[1])
                ]
                .groupby("year")
                .agg(movie_count=("movie_id", "count"), avg_rating=("avg_rating", "mean"))
                .reset_index()
            )

            # ── Row 1: Avg Rating by Year + Genre Pie ──
            col_r1a, col_r1b = st.columns(2)

            with col_r1a:
                fig8 = px.scatter(
                    year_df, x="year", y="avg_rating",
                    title=f"Avg Rating by Year ({yr_range[0]}–{yr_range[1]})",
                    labels={"year": "Release Year", "avg_rating": "Avg Rating"},
                    color_discrete_sequence=["#e50914"],
                    trendline="lowess",
                    trendline_color_override="#888",
                )
                fig8.update_layout(**_PLOT_BASE, title_font_color="#e8e8e8")
                st.plotly_chart(fig8, use_container_width=True)
                st.markdown(
                    '<div style="font-size:0.75rem;color:#555;margin-top:-0.8rem;padding-bottom:0.5rem">'
                    'Insight: Older films (pre-1980) receive higher average ratings — '
                    'survivorship bias: only highly regarded classics from that era remain in the catalog.'
                    '</div>', unsafe_allow_html=True
                )

            with col_r1b:
                _gstats_pie = _genre_stats(filt_df)
                _pie_top9   = _gstats_pie.head(9)
                _pie_other  = int(_gstats_pie["movie_count"].iloc[9:].sum())
                _pie_df = pd.concat([
                    _pie_top9[["genre", "movie_count"]].rename(columns={"movie_count": "count"}),
                    pd.DataFrame([{"genre": "Other", "count": _pie_other}]),
                ], ignore_index=True)
                fig_pie = px.pie(
                    _pie_df, values="count", names="genre",
                    title=f"Movies by Genre (min {min_ratings:,} ratings/movie)",
                    hole=0.38,
                    color_discrete_sequence=[
                        "#e50914","#c5000f","#a00010","#800020",
                        "#ff4444","#ff7777","#ffaaaa","#cc3333","#992222","#444",
                    ],
                )
                _pb_pie = {k: v for k, v in _PLOT_BASE.items() if k not in ("xaxis", "yaxis")}
                fig_pie.update_layout(
                    **_pb_pie,
                    title_font_color="#e8e8e8",
                    showlegend=True,
                    legend=dict(font=dict(color="#888", size=9)),
                )
                fig_pie.update_traces(
                    textposition="inside", textinfo="percent",
                    textfont=dict(size=9, color="#fff"),
                    marker=dict(line=dict(color="#111111", width=1)),
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            # ── Row 2: User activity + Rating longtail ──
            col_l3, col_r3 = st.columns(2)

            with col_l3:
                if users_df is not None:
                    buckets = pd.cut(
                        users_df["rating_count"],
                        bins=[0, 50, 100, 250, 500, 1000, 5000, 50000],
                        labels=["1–50", "51–100", "101–250", "251–500", "501–1K", "1K–5K", "5K+"],
                    )
                    bucket_counts = buckets.value_counts().sort_index().reset_index()
                    bucket_counts.columns = ["ratings_given", "users"]
                    bucket_counts["pct"] = (bucket_counts["users"] / bucket_counts["users"].sum() * 100).round(1)

                    fig5 = px.bar(
                        bucket_counts, x="ratings_given", y="users",
                        title="User Activity Distribution",
                        labels={"ratings_given": "Ratings Given", "users": "Number of Users"},
                        color_discrete_sequence=["#e50914"],
                        text=bucket_counts["pct"].astype(str) + "%",
                    )
                    fig5.update_layout(**_PLOT_BASE, title_font_color="#e8e8e8")
                    fig5.update_traces(marker_line_width=0, textposition="outside",
                                       textfont=dict(color="#666", size=10))
                    st.plotly_chart(fig5, use_container_width=True)
                    st.markdown(
                        '<div style="font-size:0.75rem;color:#555;margin-top:-0.8rem;padding-bottom:0.5rem">'
                        'Insight: Most users rate fewer than 100 movies — sparse interactions '
                        'make collaborative filtering challenging without the ALS latent-factor approach.'
                        '</div>', unsafe_allow_html=True
                    )

            with col_r3:
                longtail = (
                    filt_df[["movie_id", "title", "rating_count"]]
                    .sort_values("rating_count", ascending=False)
                    .reset_index(drop=True)
                )
                longtail["rank"] = longtail.index + 1
                longtail["cumulative_pct"] = (longtail["rating_count"].cumsum() / longtail["rating_count"].sum() * 100)

                top10_pct = longtail[longtail["rank"] <= int(len(longtail) * 0.1)]["cumulative_pct"].iloc[-1]

                fig6 = px.line(
                    longtail[longtail["rank"] <= min(5000, len(longtail))],
                    x="rank", y="rating_count",
                    title="The Long-Tail Problem: Movie Popularity Distribution",
                    labels={"rank": "Movie Rank (by popularity)", "rating_count": "Number of Ratings"},
                    color_discrete_sequence=["#e50914"],
                )
                fig6.update_layout(**_PLOT_BASE, title_font_color="#e8e8e8")
                fig6.add_vline(x=int(len(longtail) * 0.1), line_dash="dash",
                               line_color="#555", annotation_text="Top 10%",
                               annotation_font_color="#666")
                st.plotly_chart(fig6, use_container_width=True)
                st.markdown(
                    f'<div style="font-size:0.75rem;color:#555;margin-top:-0.8rem;padding-bottom:0.5rem">'
                    f'Insight: The top 10% of movies account for {top10_pct:.0f}% of all ratings. '
                    f'ALS collaborative filtering is essential to surface quality films in the long tail.'
                    f'</div>', unsafe_allow_html=True
                )

            # ── Row 3: Movies Per Year (full-width) ──
            fig7 = px.area(
                year_df, x="year", y="movie_count",
                title=f"Movies Per Year ({yr_range[0]}–{yr_range[1]})",
                labels={"year": "Release Year", "movie_count": "Movies"},
                color_discrete_sequence=["#e50914"],
            )
            fig7.update_layout(**_PLOT_BASE, title_font_color="#e8e8e8")
            fig7.update_traces(fillcolor="rgba(229,9,20,0.15)", line_color="#e50914")
            st.plotly_chart(fig7, use_container_width=True)

            st.markdown("<hr>", unsafe_allow_html=True)

            # ── Row 3: Rating distribution + Genre popularity ──
            col_l, col_r = st.columns(2)

            with col_l:
                if rating_dist is not None:
                    fig = px.bar(
                        rating_dist, x="rating", y="count",
                        title="Rating Distribution (25M ratings)",
                        labels={"rating": "Star Rating", "count": "Number of Ratings"},
                        color_discrete_sequence=["#e50914"],
                    )
                    fig.update_layout(**_PLOT_BASE, title_font_color="#e8e8e8")
                    fig.update_traces(marker_line_width=0)
                    fig.add_annotation(
                        text="4.0★ is the most common rating — users preferentially watch movies they expect to like",
                        xref="paper", yref="paper", x=0.5, y=1.0,
                        showarrow=False, font=dict(size=9, color="#666"),
                        xanchor="center", yanchor="bottom",
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with col_r:
                gstats = _genre_stats(filt_df).head(15)
                fig2 = px.bar(
                    gstats.sort_values("total_ratings"),
                    x="total_ratings", y="genre",
                    orientation="h",
                    title=f"Total Ratings by Genre (min {min_ratings:,} ratings/movie)",
                    labels={"total_ratings": "Ratings", "genre": ""},
                    color_discrete_sequence=["#e50914"],
                )
                fig2.update_layout(**_PLOT_BASE, title_font_color="#e8e8e8")
                fig2.update_traces(marker_line_width=0)
                st.plotly_chart(fig2, use_container_width=True)

            # ── Row 4: Genre avg rating (bias) + Top movies ──
            col_l2, col_r2 = st.columns(2)

            with col_l2:
                gstats_full = _genre_stats(filt_df)
                gstats_full = gstats_full[gstats_full["movie_count"] >= 10].copy()
                gstats_full = gstats_full.sort_values("avg_rating", ascending=True).tail(18)
                colors = ["#e50914" if r >= 3.3 else "#555" for r in gstats_full["avg_rating"]]
                fig3 = go.Figure(go.Bar(
                    x=gstats_full["avg_rating"],
                    y=gstats_full["genre"],
                    orientation="h",
                    marker_color=colors,
                    marker_line_width=0,
                    text=gstats_full["avg_rating"].round(2),
                    textposition="outside",
                    textfont=dict(color="#888", size=10),
                ))
                _pb = {k: v for k, v in _PLOT_BASE.items() if k not in ("xaxis", "yaxis")}
                fig3.update_layout(
                    **_pb,
                    title="Average Rating by Genre",
                    title_font_color="#e8e8e8",
                    xaxis=dict(range=[2.5, 3.7], gridcolor="#1e1e1e", zerolinecolor="#1e1e1e"),
                    yaxis=dict(gridcolor="#1e1e1e", zerolinecolor="#1e1e1e"),
                )
                st.plotly_chart(fig3, use_container_width=True)
                st.markdown(
                    '<div style="font-size:0.75rem;color:#555;margin-top:-0.8rem;padding-bottom:0.5rem">'
                    'Insight: Documentary, History, and War films earn systematically higher ratings. '
                    'Horror and Sci-Fi lag — genre expectations shape how users rate.'
                    '</div>', unsafe_allow_html=True
                )

            with col_r2:
                top_movies = (
                    filt_df
                    .sort_values("rating_count", ascending=False)
                    .head(top_n)
                    .sort_values("rating_count")
                )
                fig4 = px.bar(
                    top_movies, x="rating_count", y="title",
                    orientation="h",
                    title=f"Top {top_n} Most-Rated Movies (min {min_ratings:,} ratings)",
                    labels={"rating_count": "Ratings", "title": ""},
                    color_discrete_sequence=["#c5000f"],
                )
                fig4.update_layout(**_PLOT_BASE, title_font_color="#e8e8e8",
                                   height=max(340, top_n * 22))
                fig4.update_traces(marker_line_width=0)
                st.plotly_chart(fig4, use_container_width=True)
