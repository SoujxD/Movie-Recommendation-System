# Movie Recommendation System 🎬✨

## Overview 📊
Welcome to the **Movie Recommendation System**! 🚀 This project leverages advanced techniques in natural language processing (NLP) and machine learning to provide users with personalized movie recommendations based on their input description or keywords. The system combines the power of **DistilBERT embeddings**, **TF-IDF vectorization**, **KMeans clustering**, and **diversity penalties** to create a more tailored and diverse movie list for each user. 

## Key Features 🔑
- **DistilBERT-based Movie Embeddings**: We use DistilBERT to generate semantic embeddings for movie descriptions, capturing the essence of each movie's plot, genre, and more.
- **TF-IDF Vectorization**: This technique ensures that the recommended movies are not only similar to the user input but also diverse. The system accounts for both textual similarity and keyword importance, offering recommendations with a balance of relevance and variety.
- **KMeans Clustering**: Movies are grouped into clusters based on their content, allowing us to provide recommendations from similar genres and styles.
- **Diversity Penalty**: To prevent redundant suggestions, a penalty is applied to movies too similar to the user's query, ensuring diverse recommendations. This promotes a variety of choices and offers a richer movie-watching experience.

## Why This Project is Different 🚀
This movie recommendation system stands out because of the innovative combination of several techniques that go beyond simple content-based filtering:
- **TF-IDF for Enhanced Diversity**: By using **TF-IDF vectorization**, the model goes a step further to account for not only the similarity between movie descriptions but also the importance of keywords. This ensures that the system doesn't just recommend movies that are similar to the input but also considers the diversity of the movies, avoiding monotony in suggestions.
- **KMeans Clustering for Contextual Relevance**: By grouping movies into clusters, we can ensure that recommendations come from a pool of movies that share contextual and thematic similarities, improving the relevance and quality of recommendations.
- **Diversity Penalty for Balanced Suggestions**: The **diversity penalty** is a unique feature of this project. It adjusts the similarity scores to encourage variety in the recommendations. This penalty prevents the system from suggesting movies that are too similar, ensuring that users get a fresh and diverse set of suggestions.

## Technologies Used 💻
- **Python**: The backbone of this project, enabling smooth implementation of machine learning and NLP techniques.
- **DistilBERT**: Used for generating semantic embeddings of movie descriptions, providing an in-depth understanding of movie content.
- **Scikit-learn**: Used for clustering movies with **KMeans** and for vectorization using **TF-IDF**.
- **Streamlit**: Used to build an interactive web interface for users to interact with the recommendation system.

## How It Works 🔄
1. **Data Collection**: Movie data (from TMDB) is collected, cleaned, and processed to create a comprehensive dataset of movie details.
2. **Movie Embeddings Generation**: Using DistilBERT, we generate embeddings for movie descriptions that capture deep semantic meaning.
3. **Clustering**: Movies are grouped into clusters using **KMeans**, helping to categorize movies based on their content.
4. **User Input**: Users input a description or keywords of movies they like or are interested in.
5. **Recommendation Engine**: The system computes recommendations by comparing the user input with precomputed embeddings and applying the **TF-IDF similarity** and **diversity penalty** to provide relevant and diverse suggestions.

## Example 🎥

### User Input:
- **Query**: "Action thriller with a strong plot and intense chase scenes"

### Output Recommendations:
- **Movie 1**: Title, Release Year, Genres, Similarity Score, Popularity, Vote Average
- **Movie 2**: Title, Release Year, Genres, Similarity Score, Popularity, Vote Average
- *(and so on)*

## Conclusion 🎯
The **Enhanced Movie Recommendation System** takes personalized movie suggestions to the next level by incorporating **TF-IDF for keyword relevance**, **KMeans clustering for thematic grouping**, and a **diversity penalty** to ensure a well-rounded recommendation list. Whether you're looking for similar films or something entirely different, this system provides both accuracy and variety. Happy movie watching! 🍿🎉

---


