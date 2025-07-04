import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


ratings_df = pd.read_csv("dataset/ratings.csv")
movies_df = pd.read_csv("dataset/movies.csv")


ratings_df.drop(columns=["timestamp"], inplace=True)# Dropping Timestamps

# user-item matrix
user_item_matrix = ratings_df.pivot_table(index='userId', columns='movieId', values='rating')
user_item_matrix.fillna(0, inplace=True)

# Similarity calculation
user_sim = cosine_similarity(user_item_matrix)
user_sim_df = pd.DataFrame(user_sim, index=user_item_matrix.index, columns=user_item_matrix.index)

# User input
movie_title_input = input(" Enter a movie title to predict rating for: ").strip().lower()
user_id_input = int(input(" Enter user ID to predict for: "))

# Find movie ID
matched = movies_df[movies_df['title'].str.lower().str.contains(movie_title_input)]
if matched.empty:
    print("Error: Movie not found.")
else:
    movie_id = matched.iloc[0]['movieId']
    movie_title = matched.iloc[0]['title']

    # Ratings by all users
    movie_ratings = user_item_matrix[movie_id]
    similarities = user_sim_df[user_id_input]

    # Predicted rating 
    weighted_sum = sum(similarities * movie_ratings)
    sim_sum = sum(similarities)
    predicted_rating = weighted_sum / sim_sum if sim_sum != 0 else 0

    print(f"\n Predicted rating for user {user_id_input} on '{movie_title}' is: {predicted_rating:.2f}")

# Movie-to-Movie similarity
item_user_matrix = ratings_df.pivot_table(index='movieId', columns='userId', values='rating')
item_user_matrix.fillna(0, inplace=True)

movie_sim = cosine_similarity(item_user_matrix)
movie_sim_df = pd.DataFrame(movie_sim, index=item_user_matrix.index, columns=item_user_matrix.index)

# top 5 similar movies
similar_scores = movie_sim_df[movie_id].sort_values(ascending=False)[1:6]
similar_df = pd.DataFrame({'movieId': similar_scores.index, 'score': similar_scores.values})
similar_named = pd.merge(similar_df, movies_df, on='movieId')

print(f"\n Movies similar to '{movie_title}':\n")
for _, row in similar_named.iterrows():
    print(f"- {row['title']} → similarity score: {row['score']:.2f}")
