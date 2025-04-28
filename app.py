import pandas as pd
import streamlit as st
import logging
import os
import time

# Set up logging to debug performance issues
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Streamlit page configuration
st.set_page_config(page_title="Movie Recommendation System", page_icon="🎬", layout="centered")

# Show a loading message while the app initializes
with st.spinner("Initializing app and loading data..."):
    # Cache the data loading and processing to improve performance
    @st.cache_data
    def load_and_process_data():
        start_time = time.time()
        try:
            data_path = r'\movie_recommendation_system\merged_dataset.csv'
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Dataset file not found at: {data_path}")
            
            logger.info("Loading dataset...")
            merged_data = pd.read_csv(data_path)
            logger.info(f"Dataset loaded in {time.time() - start_time:.2f} seconds")

            # Define the list of possible genres
            genre_cols = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 
                          'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

            # Check if 'genres' column exists
            if 'genres' not in merged_data.columns:
                raise ValueError("The 'genres' column is not found in merged_data. Please ensure the dataset includes genre information.")
            
            start_time = time.time()
            logger.info("Processing genres...")
            # Function to split pipe-separated genres and create binary columns
            def process_genres(df, genre_list):
                for genre in genre_list:
                    df[genre] = 0
                df['genres'] = df['genres'].fillna('')
                for genre in genre_list:
                    df[genre] = df['genres'].str.contains(genre, na=False).astype(int)
                return df

            # Process genres in merged_data
            merged_data = process_genres(merged_data, genre_cols)
            logger.info(f"Genres processed in {time.time() - start_time:.2f} seconds")

            start_time = time.time()
            logger.info("Calculating average ratings...")
            # Calculate average ratings for each movie
            average_ratings = merged_data.groupby('movieId')['rating'].mean().reset_index()
            # Merge average ratings back into merged_data
            merged_data = merged_data.merge(average_ratings, on='movieId', how='left', suffixes=('', '_avg'))
            logger.info(f"Average ratings calculated in {time.time() - start_time:.2f} seconds")
            
            return merged_data, genre_cols
        except Exception as e:
            logger.error(f"Error loading or processing data: {str(e)}")
            st.error(f"Error loading data: {str(e)}")
            st.stop()

    # Load and process data
    start_time = time.time()
    try:
        merged_data, genre_cols = load_and_process_data()
        logger.info(f"Total data loading and processing took {time.time() - start_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        st.stop()

# Cache the recommendation function to avoid recomputation
@st.cache_data
def get_top_rated_movies_by_genre(genre, min_rating, top_n=5, _merged_data=None):
    start_time = time.time()
    try:
        # Filter movies by the selected genre
        genre_movies = _merged_data[_merged_data[genre] == 1]
        if genre_movies.empty:
            return pd.DataFrame({'title': [f"No movies found for genre: {genre}"]})
        
        # Filter movies by minimum average rating
        genre_movies = genre_movies[genre_movies['rating_avg'] >= min_rating]
        if genre_movies.empty:
            return pd.DataFrame({'title': [f"No movies found for genre {genre} with average rating >= {min_rating}"]})
        
        # Group by movieId and title, sort by average rating, and take top N
        top_movies = (genre_movies.groupby(['movieId', 'title'])['rating_avg']
                      .mean()
                      .reset_index()
                      .sort_values(by='rating_avg', ascending=False)
                      .head(top_n))
        
        logger.info(f"Recommendations generated in {time.time() - start_time:.2f} seconds")
        return top_movies[['title']]
    except Exception as e:
        logger.error(f"Error in recommendation: {str(e)}")
        return pd.DataFrame({'title': [f"Error generating recommendations: {str(e)}"]})

# Streamlit UI
st.title("🎬 Movie Recommendation System")
st.markdown("Discover the best movies in your favorite genre based on average ratings!")

# Create a container for the input form
with st.container():
    st.subheader("Select Your Preferences")
    
    # Genre selection
    genre = st.selectbox("Select Genre:", options=genre_cols)
    
    # Minimum rating input
    min_rating = st.number_input("Minimum Average Rating (0-5):", min_value=0.0, max_value=5.0, value=3.0, step=0.1)
    
    # Button to get recommendations
    if st.button("Get Recommendations"):
        with st.spinner("Fetching recommendations..."):
            recommendations = get_top_rated_movies_by_genre(genre, min_rating, _merged_data=merged_data)
            recommendations_list = recommendations['title'].tolist()
        
        # Display recommendations
        st.subheader("Recommended Movies")
        if "No movies found" in recommendations_list[0] or "Error" in recommendations_list[0]:
            st.warning(recommendations_list[0])
        else:
            for i, movie in enumerate(recommendations_list, 1):
                st.write(f"{i}. {movie}")

# Add some custom CSS for styling
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
