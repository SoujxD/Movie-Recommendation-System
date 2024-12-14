import pandas as pd
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.cluster import KMeans
import torch
from sklearn.preprocessing import MinMaxScaler

# Load the data
df = pd.read_csv("C:/MLprojects/movierecommendation/tmdb_5000_movies.csv")
df = df.dropna(subset=['overview', 'vote_average', 'release_date', 'popularity'])
df['description'] = df['title'] + ' ' + df['genres'] + ' ' + df['overview']
df['release_date'] = pd.to_datetime(df['release_date']).dt.year
sc = MinMaxScaler()
df[['popularity', 'vote_average']] = sc.fit_transform(df[['popularity', 'vote_average']])

# Load tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Generate embeddings
def get_embeddings(text, batch_size=32):
    embeddings = []
    for i in range(0, len(text), batch_size):
        batch_text = text[i:i + batch_size]
        encoded_text = tokenizer(
            batch_text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )
        with torch.no_grad():
            output = model(**encoded_text)
        batch_embeddings = output.last_hidden_state.mean(dim=1)
        embeddings.append(batch_embeddings)
    return torch.cat(embeddings)

movie_embeddings = get_embeddings(df['description'].tolist())

# KMeans clustering
kmeans = KMeans(n_clusters=10, random_state=42)
df['cluster'] = kmeans.fit_predict(movie_embeddings.numpy())

# Save embeddings and clusters
torch.save(movie_embeddings, "C:/MLprojects/movierecommendation/movie_embeddings.pt")
cluster_df = pd.DataFrame({'cluster': df['cluster']})
cluster_df.to_csv("C:/MLprojects/movierecommendation/movie_clusters.csv", index=False)

print("Precomputation complete. Embeddings and clusters saved.")
