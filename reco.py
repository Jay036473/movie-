import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.metrics import pairwise_distances
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings("ignore")

@st.cache_data
def load_data():
    rating = pd.read_csv("rating.csv")
    movies = pd.read_csv("movie.csv")
    merge = pd.merge(rating, movies, on="movieId", how="left")
    df = merge[["userId", "title", "rating"]]
    return df
df = load_data()

#st.dataframe(df)

def popularity_recommender(df, top_n=10):
    #popular_moives = df.groupby('title')['rating'].mean().sort_values(ascending=False).head(top_n)
    df1 = df.groupby('title').agg({'rating':'mean','userId':'count'}).reset_index()#.sort_values(ascending=False).head(top_n)
    df2 = df1.sort_values(by=['userId','rating'],ascending=[False,False]).head(top_n)
    return df2

def item_based_recommender(Name, top_n=6):
    ratings_matrix = df.pivot_table(index='userId', columns='title', values='rating').fillna(0)
    user_matrix = ratings_matrix.T
    cos_sim = cosine_similarity(user_matrix)
    cos_sim_df = pd.DataFrame(cos_sim, index=user_matrix.index, columns=user_matrix.index)
    if Name in cos_sim_df:
        #return cos_sim_df[Name].sort_values(ascending=False)[1:top_n+1]
        return cos_sim_df[Name].sort_values(ascending=False).head(10)
    else:
        return "Moive is not in dataset"

def user_base_rec(User,n_neighbors=6):
    ratings_matrix = df.pivot_table(index='userId', columns='title', values='rating').fillna(0)
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(ratings_matrix)
    if User in ratings_matrix.index:
        distances, indices = model_knn.kneighbors([ratings_matrix.loc[User]], n_neighbors=n_neighbors + 1)
        #similar_users = ratings_matrix.index[indices.flatten()[1:]]
        similar_users = ratings_matrix.index[indices.flatten()[1:]]
        return similar_users
    else:
        return "User not in dataset"

b1 = st.sidebar.checkbox("uniqe Value")
if b1:
    st.dataframe(df["rating"].value_counts())
    for i in df.columns:
        st.write("The dataset has", len(df[i].unique()), "Unique", i)

b2 = st.sidebar.checkbox("Matrix")
if b2:
    ratings_matrix = df.pivot_table(index='userId', columns='title', values='rating').fillna(0)
    st.dataframe(ratings_matrix.iloc[1:1050,0:20000])

b3 = st.sidebar.checkbox("Popular Movies")
if b3:
    st.write("\nPopular Movies:\n", popularity_recommender(df))

b3 = st.sidebar.checkbox("Items Movies")
if b3:
    Name = st.selectbox("Moive Name",df['title'].unique())
    st.write("\nItems base Moives:\n",item_based_recommender(Name))

b4 = st.sidebar.checkbox("User Base Moives")
if b4:
    User = st.number_input("User ID",1,100000,5)
    df1 = user_base_rec(User)
    new = df[df['userId'].isin(df1)].sort_values(by='rating',ascending=False).head(10)
    new1 = new['title'].unique()

    st.dataframe(new1)




