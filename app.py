import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.feature_extraction.text import TfidfVectorizer

# Title
st.title("Enhanced Movie Recommendation System")
st.write("Get diverse movie recommendations based on a description or keywords!")

# Load the data
@st.cache_data
def load_data():
    df = pd.read_csv("C:/MLprojects/movierecommendation/tmdb_5000_movies.csv")
    df = df.dropna(subset=['overview', 'vote_average', 'release_date', 'popularity'])
    df['description'] = df['title'] + ' ' + df['genres'] + ' ' + df['overview']
    df['release_date'] = pd.to_datetime(df['release_date']).dt.year
    return df

df = load_data()

# Load precomputed embeddings and clusters
@st.cache_data
def load_precomputed_data():
    embeddings = torch.load("C:/MLprojects/movierecommendation/movie_embeddings.pt")
    clusters = pd.read_csv("C:/MLprojects/movierecommendation/movie_clusters.csv")
    return embeddings, clusters

movie_embeddings, cluster_data = load_precomputed_data()
df['cluster'] = cluster_data['cluster']

# Load DistilBERT tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Recommendation function
def get_recommendations(query, embeddings, df, top_n=5, filter_year=None):
    # Compute query embeddings without caching
    def compute_query_embeddings(query_text):
        encoded_text = tokenizer(
            [query_text],
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )
        with torch.no_grad():
            output = model(**encoded_text)
        return output.last_hidden_state.mean(dim=1)
    
    query_embeddings = compute_query_embeddings(query)
    similarity = cosine_similarity(query_embeddings.numpy(), embeddings.numpy())
    
    df['similarity'] = similarity[0]
    
    # TF-IDF vectorization for diversity
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['description'])
    query_tfidf = tfidf.transform([query])
    tfidf_similarity = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
    
    df['tfidf_similarity'] = tfidf_similarity

    # Apply filters
    filtered_df = df.copy()
    if filter_year:
        filtered_df = filtered_df[filtered_df['release_date'] == filter_year]

    # Compute weighted score with diversity
    filtered_df['weighted_score'] = (
        filtered_df['popularity'] * 0.3 +
        filtered_df['vote_average'] * 0.2 +
        filtered_df['similarity'] * 0.3 +
        filtered_df['tfidf_similarity'] * 0.2
    )

    # Penalize similarity for diversity
    filtered_df['diversity_penalty'] = filtered_df['similarity'] * -0.1
    filtered_df['final_score'] = filtered_df['weighted_score'] + filtered_df['diversity_penalty']
    
    return filtered_df.sort_values(by='final_score', ascending=False).head(top_n)

# User inputs
query = st.text_input("Enter a movie description or keywords:")
filter_year = st.selectbox("Filter by release year (Recommended):", [None] + sorted(df['release_date'].dropna().unique().tolist()))

top_n = st.slider("Number of recommendations:", min_value=1, max_value=20, value=5)

# Generate recommendations
if st.button("Get Recommendations"):
    if not query.strip():
        st.error("Please enter a movie description or keywords!")
    else:
        recommendations = get_recommendations(query, movie_embeddings, df, top_n=top_n, filter_year=filter_year)
        
        if recommendations.empty:
            st.write("No recommendations found for the given filters.")
        else:
            st.write("### Recommendations:")
            for _, row in recommendations.iterrows():
                st.subheader(row['title'])
                st.write(f"**Release Year:** {row['release_date']}")
                st.write(f"**Genres:** {row['genres']}")
                st.write(f"**Similarity Score:** {row['similarity']:.2f}")
                st.write(f"**Popularity:** {row['popularity']:.2f}")
                st.write(f"**Vote Average:** {row['vote_average']:.2f}")
                st.write("---")
