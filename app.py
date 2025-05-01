import pandas as pd
import streamlit as st
import logging
import os
import time
from kagglehub import dataset_download

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(page_title="Movie Recommendation System", page_icon="🎬", layout="centered")

# Load dataset
@st.cache_data
def load_data():
    logger.info("Downloading dataset from Kaggle...")
    path = dataset_download("namanjha4050/movie-recommendation")
    csv_path = os.path.join(path, "merged_dataset.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    return df

# Process genres
@st.cache_data
def process_genres(df):
    genre_cols = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
                  'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
                  'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

    if 'genres' not in df.columns:
        raise ValueError("Column 'genres' is missing from the dataset.")

    df['genres'] = df['genres'].fillna('')

    for genre in genre_cols:
        df[genre] = df['genres'].str.contains(genre, na=False).astype(int)

    return df, genre_cols

# Calculate average ratings
@st.cache_data
def calculate_avg_ratings(df):
    avg_ratings = df.groupby('movieId')['rating'].mean().reset_index()
    df = df.merge(avg_ratings.rename(columns={'rating': 'rating_avg'}), on='movieId', how='left')
    return df

# Get top rated movies by genre
@st.cache_data
def get_top_rated_movies_by_genre(genre, min_rating, _merged_data=None, top_n=5):
    try:
        genre_movies = _merged_data[_merged_data[genre] == 1]
        if genre_movies.empty:
            return pd.DataFrame({'title': [f"No movies found for genre: {genre}"]})
        
        genre_movies = genre_movies[genre_movies['rating_avg'] >= min_rating]
        if genre_movies.empty:
            return pd.DataFrame({'title': [f"No movies found for genre '{genre}' with rating ≥ {min_rating}"]})

        top_movies = (
            genre_movies.groupby(['movieId', 'title'])['rating_avg']
            .mean()
            .reset_index()
            .sort_values(by='rating_avg', ascending=False)
            .head(top_n)
        )
        return top_movies[['title']]
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        return pd.DataFrame({'title': [f"Error: {str(e)}"]})

# Main app logic
def main():
    # Load and process data
    with st.spinner("Initializing and loading data..."):
        try:
            merged_data = load_data()
            merged_data, genre_cols = process_genres(merged_data)
            merged_data = calculate_avg_ratings(merged_data)
        except Exception as e:
            st.error(f"Failed to initialize: {e}")
            logger.error(f"Initialization failed: {e}")
            return

    # UI Elements
    st.title("🎬 Movie Recommendation System")
    st.markdown("Discover the best movies in your favorite genre based on average ratings!")

    with st.container():
        st.subheader("Select Your Preferences")
        genre = st.selectbox("Select Genre:", options=genre_cols)
        min_rating = st.slider("Minimum Average Rating (0-5):", min_value=0.0, max_value=5.0, value=3.0, step=0.1)

        if st.button("Get Recommendations"):
            with st.spinner("Fetching recommendations..."):
                recommendations = get_top_rated_movies_by_genre(genre, min_rating, _merged_data=merged_data)

            st.subheader("Recommended Movies")
            rec_list = recommendations['title'].tolist()

            if "No movies found" in rec_list[0] or "Error" in rec_list[0]:
                st.warning(rec_list[0])
            else:
                for i, movie in enumerate(rec_list, 1):
                    st.write(f"{i}. {movie}")

    # Custom Styling
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(120deg, #ff8c00, #f97316);
        padding: 20px;
    }
    .stButton > button {
        background: linear-gradient(90deg, #ff8c00, #f97316);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #e67e22, #d65c14);
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(249, 115, 22, 0.4);
    }
    .stSelectbox, .stNumberInput {
        background: #f7f9fc;
        border-radius: 10px;
        padding: 5px;
    }
    .stSelectbox > div > div, .stNumberInput > div > div {
        border: 2px solid #dfe6e9 !important;
        border-radius: 10px !important;
    }
    h1, h2 {
        color: #2c3e50 !important;
        text-align: center;
    }
    .stMarkdown {
        text-align: center;
        color: #34495e;
    }
    </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
