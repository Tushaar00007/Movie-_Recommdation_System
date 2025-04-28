import pandas as pd

# Load the datasets
df1 = pd.read_csv(r'C:\AIML\movie_recommendation_system\moviedataset\movies.csv')
df2 = pd.read_csv(r'C:\AIML\movie_recommendation_system\moviedataset\ratings.csv')

# Merge the datasets on the correct column
merged_df = pd.merge(df1, df2, on='movieId')  # Replace 'movieId' with the actual column name

# Save the merged dataset
merged_df.to_csv(r'C:\AIML\movie_recommendation_system\moviedataset\merged_dataset.csv', index=False)

print("Datasets merged successfully!")