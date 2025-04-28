import pandas as pd
from sklearn.model_selection import train_test_split
from numpy import sqrt
from sklearn.metrics import mean_squared_error
import scipy.sparse as sparse
import numpy as np
from scipy.sparse.linalg import svds

# Load the merged dataset
merged_data = pd.read_csv(r'\movie_recommendation_system\merged_dataset.csv')

# Basic exploration
print("Merged data head:\n", merged_data.head())
print("Number of users:", merged_data['userId'].nunique())
print("Number of movies:", merged_data['movieId'].nunique())

# Filter users and movies with at least 100 ratings to reduce dataset size
min_ratings = 100
user_counts = merged_data['userId'].value_counts()
movie_counts = merged_data['movieId'].value_counts()
filtered_data = merged_data[merged_data['userId'].isin(user_counts[user_counts >= min_ratings].index)]
filtered_data = filtered_data[filtered_data['movieId'].isin(movie_counts[movie_counts >= min_ratings].index)]
print("Filtered data shape:", filtered_data.shape)
print("Number of users after filtering:", filtered_data['userId'].nunique())
print("Number of movies after filtering:", filtered_data['movieId'].nunique())

# Create a sparse user-item matrix
user_ids = filtered_data['userId'].astype('category').cat.codes
movie_ids = filtered_data['movieId'].astype('category').cat.codes
ratings = filtered_data['rating'].astype(np.float32)
user_item_matrix = sparse.csr_matrix((ratings, (user_ids, movie_ids)),
                         shape=(user_ids.max() + 1, movie_ids.max() + 1))
print("Sparse user-item matrix shape:", user_item_matrix.shape)

# Perform SVD to get user and movie embeddings
k = 50  # Number of latent factors (embedding dimensions)
U, sigma, Vt = svds(user_item_matrix, k=k)
sigma = np.diag(sigma)
user_embeddings = U  # Shape: (n_users, k)
movie_embeddings = Vt.T  # Shape: (n_movies, k)
print("User embeddings shape:", user_embeddings.shape)
print("Movie embeddings shape:", movie_embeddings.shape)

def get_user_recommendations(user_id, user_embeddings, movie_embeddings, merged_data, user_ids_map, movie_ids_map, top_n=5):
    # Map user_id to index
    if user_id not in user_ids_map:
        return pd.DataFrame({'title': ["User not found"]})
    user_idx = user_ids_map[user_id]
    
    # Compute predicted ratings for all movies using dot product of embeddings
    predicted_ratings = np.dot(user_embeddings[user_idx], movie_embeddings.T)
    
    # Get movies the user has already rated
    user_rated_indices = user_item_matrix[user_idx, :].nonzero()[1]
    
    # Set predicted ratings for already rated movies to -inf to exclude them
    predicted_ratings[user_rated_indices] = -np.inf
    
    # Get top N movie indices
    top_movie_indices = np.argsort(predicted_ratings)[::-1][:top_n]
    
    # Map indices back to movie IDs
    movie_id_map = dict(enumerate(filtered_data['movieId'].astype('category').cat.categories))
    recommended_movie_ids = [movie_id_map[idx] for idx in top_movie_indices]
    
    # Map movie IDs to titles
    recommended_movies = merged_data[merged_data['movieId'].isin(recommended_movie_ids)][['title']].drop_duplicates()
    return recommended_movies

# Create a mapping of original user IDs and movie IDs to indices
user_ids_map = dict(zip(filtered_data['userId'], user_ids))
movie_ids_map = dict(zip(filtered_data['movieId'], movie_ids))

# Test the recommender
test_user_id = 1
recommendations = get_user_recommendations(test_user_id, user_embeddings, movie_embeddings, merged_data, user_ids_map, movie_ids_map)
print(f"Recommendations for user {test_user_id}:\n", recommendations)

# Split data for evaluation
train_data, test_data = train_test_split(filtered_data, test_size=0.2, random_state=42)
print("Train data shape:", train_data.shape)
print("Test data shape:", test_data.shape)

# Create training sparse user-item matrix
train_user_ids = train_data['userId'].astype('category').cat.codes
train_movie_ids = train_data['movieId'].astype('category').cat.codes
train_ratings = train_data['rating'].astype(np.float32)
train_user_item_matrix = sparse.csr_matrix((train_ratings, (train_user_ids, train_movie_ids)),
                                 shape=(train_user_ids.max() + 1, train_movie_ids.max() + 1))

# Perform SVD on training data
U, sigma, Vt = svds(train_user_item_matrix, k=k)
sigma = np.diag(sigma)
train_user_embeddings = U
train_movie_embeddings = Vt.T

def predict_rating(user_id, movie_id, user_embeddings, movie_embeddings, user_ids_map, movie_ids_map):
    if user_id not in user_ids_map or movie_id not in movie_ids_map:
        return 0
    user_idx = user_ids_map[user_id]
    movie_idx = movie_ids_map[movie_id]
    predicted_rating = np.dot(user_embeddings[user_idx], movie_embeddings[movie_idx])
    return predicted_rating if predicted_rating > 0 else 0

# Create mappings for train data
train_user_ids_map = dict(zip(train_data['userId'], train_user_ids))
train_movie_ids_map = dict(zip(train_data['movieId'], train_movie_ids))

# Predict ratings for test set
test_predictions = []
test_actual = test_data['rating'].values.tolist()
for _, row in test_data.iterrows():
    pred = predict_rating(row['userId'], row['movieId'], train_user_embeddings, train_movie_embeddings,
                         train_user_ids_map, train_movie_ids_map)
    test_predictions.append(pred)

# Debug lengths before filtering
print("Length of test_actual before filtering:", len(test_actual))
print("Length of test_predictions before filtering:", len(test_predictions))

# Filter both test_actual and test_predictions to include only pairs where prediction > 0
filtered_pairs = [(actual, pred) for actual, pred in zip(test_actual, test_predictions) if pred > 0]
if not filtered_pairs:
    print("No valid predictions (all predictions are 0). Cannot compute RMSE.")
else:
    filtered_actual, filtered_predictions = zip(*filtered_pairs)
    print("Length of filtered_actual:", len(filtered_actual))
    print("Length of filtered_predictions:", len(filtered_predictions))

    # Compute RMSE
    mse = mean_squared_error(filtered_actual, filtered_predictions)
    rmse = sqrt(mse)
    print(f"RMSE: {rmse}")
