import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Movie Recommender", layout="wide")

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #020617, #0f172a, #020617);
    color: #e5e7eb;
}

section[data-testid="stSidebar"] {
    background: rgba(2, 6, 23, 0.85);
    backdrop-filter: blur(14px);
    border-right: 1px solid rgba(255,255,255,0.05);
}

section[data-testid="stSidebar"] * {
    color: #e5e7eb !important;
}

h1 {
    font-size: 42px;
    font-weight: 700;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
}

h2, h3 {
    color: #93c5fd;
    font-weight: 600;
}

.stButton > button {
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    color: white;
    border-radius: 14px;
    padding: 0.7em 1.6em;
    border: none;
    font-size: 15px;
    font-weight: 600;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-2px) scale(1.03);
    box-shadow: 0 10px 30px rgba(59,130,246,0.4);
}

div[data-baseweb="select"] > div,
input,
textarea {
    background: rgba(2, 6, 23, 0.7) !important;
    border-radius: 14px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    color: white !important;
}

[data-testid="stDataFrame"] {
    background: rgba(2, 6, 23, 0.65);
    border-radius: 18px;
    padding: 14px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
    backdrop-filter: blur(14px);
}

.stCheckbox > label {
    font-size: 16px;
    font-weight: 500;
}

hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, #38bdf8, transparent);
}

::-webkit-scrollbar {
    width: 7px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(#38bdf8, #818cf8);
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


st.markdown("<h1 style='text-align:center;'>🎬 Movie Recommendation System</h1>", unsafe_allow_html=True)

@st.cache_data
def load_data():
    rating = pd.read_csv("rating.csv")
    movies = pd.read_csv("movie.csv")
    merge = pd.merge(rating, movies, on="movieId", how="left")
    df = merge[["userId", "title", "rating"]]
    return df

df = load_data()

def popularity_recommender(df, top_n=10):
    df1 = df.groupby('title').agg({'rating':'mean','userId':'count'}).reset_index()
    df2 = df1.sort_values(by=['userId','rating'], ascending=[False,False]).head(top_n)
    return df2

def item_based_recommender(movie_name, top_n=6):
    ratings_matrix = df.pivot_table(index='userId', columns='title', values='rating').fillna(0)
    movie_matrix = ratings_matrix.T
    cos_sim = cosine_similarity(movie_matrix)
    cos_sim_df = pd.DataFrame(cos_sim, index=movie_matrix.index, columns=movie_matrix.index)

    if movie_name in cos_sim_df:
        return cos_sim_df[movie_name].sort_values(ascending=False)[1:top_n+1]
    else:
        return "Movie not found"

def user_based_recommender(user_id, n_neighbors=6):
    ratings_matrix = df.pivot_table(index='userId', columns='title', values='rating').fillna(0)
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(ratings_matrix)

    if user_id in ratings_matrix.index:
        distances, indices = model.kneighbors(
            [ratings_matrix.loc[user_id]],
            n_neighbors=n_neighbors + 1
        )
        similar_users = ratings_matrix.index[indices.flatten()[1:]]
        return similar_users
    else:
        return "User not found"

st.sidebar.header("🔍 Options")

b1 = st.sidebar.checkbox("Dataset Info")
if b1:
    st.subheader("Unique Ratings")
    st.dataframe(df["rating"].value_counts())
    for col in df.columns:
        st.write(f"{col} unique values:", df[col].nunique())

b2 = st.sidebar.checkbox("Rating Matrix")
if b2:
    matrix = df.pivot_table(index='userId', columns='title', values='rating').fillna(0)
    st.dataframe(matrix.iloc[:500, :50])

b3 = st.sidebar.checkbox("Popular Movies")
if b3:
    st.subheader("🔥 Top Popular Movies")
    st.dataframe(popularity_recommender(df))

b4 = st.sidebar.checkbox("Item Based Recommendation")
if b4:
    movie = st.selectbox("Select Movie", df['title'].unique())
    result = item_based_recommender(movie)
    st.subheader("🎥 Similar Movies")
    st.write(result)

b5 = st.sidebar.checkbox("User Based Recommendation")
if b5:
    user = st.number_input("Enter User ID", min_value=1, max_value=int(df.userId.max()))
    users = user_based_recommender(user)

    if isinstance(users, str):
        st.error(users)
    else:
        rec = df[df['userId'].isin(users)].sort_values(by='rating', ascending=False).head(10)
        st.subheader("👤 Recommended Movies")
        st.dataframe(rec[['title','rating']].drop_duplicates())


