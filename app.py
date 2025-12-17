import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# ============================================================
# 1. LOAD DATA
# ============================================================

movies = pd.read_csv("/content/sample_data/movies.csv")
ratings = pd.read_csv("/content/sample_data/ratings.csv")

# ============================================================
# 2. DATA PREPROCESSING
# ============================================================

# Create movie-user rating matrix
final_dataset = ratings.pivot(
    index="movieId",
    columns="userId",
    values="rating"
)

# Replace missing values with 0
final_dataset.fillna(0, inplace=True)

# Count number of ratings per movie and per user
movie_rating_count = ratings.groupby("movieId")["rating"].count()
user_rating_count = ratings.groupby("userId")["rating"].count()

# Filter movies and users with sufficient ratings
final_dataset = final_dataset.loc[
    movie_rating_count[movie_rating_count > 10].index, :
]

final_dataset = final_dataset.loc[
    :, user_rating_count[user_rating_count > 5].index
]

# Convert to sparse matrix for efficiency
csr_data = csr_matrix(final_dataset.values)

# Reset index to keep movieId as a column
final_dataset.reset_index(inplace=True)

# ============================================================
# 3. TRAIN KNN MODEL
# ============================================================

knn = NearestNeighbors(
    metric="cosine",
    algorithm="brute",
    n_neighbors=20,
    n_jobs=-1
)

knn.fit(csr_data)

# ============================================================
# 4. TOP-RATED MOVIES FUNCTION (Cold Start Support)
# ============================================================

def get_top_rated_movies_list(n=10):
    """
    Returns a markdown-formatted list of top-rated movies released after 2000.
    """

    # Average rating per movie
    avg_ratings = ratings.groupby("movieId")["rating"].mean().reset_index()

    # Rating count per movie
    rating_counts = ratings.groupby("movieId")["rating"].count().reset_index()
    rating_counts.rename(columns={"rating": "rating_count"}, inplace=True)

    # Merge datasets
    movies_with_ratings = pd.merge(avg_ratings, movies, on="movieId")
    movies_with_ratings = pd.merge(
        movies_with_ratings, rating_counts, on="movieId"
    )

    # Extract year from title
    movies_with_ratings["year"] = movies_with_ratings["title"].str.extract(
        r"\((\d{4})\)"
    ).astype(float)

    # Filter movies released after 2000
    movies_filtered = movies_with_ratings[
        (movies_with_ratings["year"] > 2000) &
        (movies_with_ratings["rating_count"] >= 10) &
        (movies_with_ratings["movieId"].isin(final_dataset["movieId"]))
    ]

    # Sort by average rating
    top_movies = movies_filtered.sort_values(
        by="rating", ascending=False
    ).head(n)

    # Format output
    result = "### Top-Rated Movies (After 2000)\n"
    for _, row in top_movies.iterrows():
        stars = "⭐" * int(round(row["rating"]))
        result += f"- **{row['title']}** | {row['rating']:.1f} {stars}\n"

    return result

# Pre-generate top movies list
top_movies_description = get_top_rated_movies_list(n=10)

# ============================================================
# 5. MOVIE RECOMMENDATION FUNCTION
# ============================================================

def get_recommendation(movie_name):
    """
    Returns a DataFrame of recommended movies similar to the input movie.
    """

    # Find movie matching input name
    movie_list = movies[
        movies["title"].str.lower().str.contains(movie_name.lower())
    ]

    if movie_list.empty:
        return "Movie not found. Please try another title."

    movie_id = movie_list.iloc[0]["movieId"]

    if movie_id not in final_dataset["movieId"].values:
        return "Not enough ratings available for this movie."

    # Get index of movie in final dataset
    movie_index = final_dataset[
        final_dataset["movieId"] == movie_id
    ].index[0]

    # Find nearest neighbors
    distances, indices = knn.kneighbors(
        csr_data[movie_index], n_neighbors=11
    )

    recommendations = []

    for idx, dist in zip(indices.squeeze(), distances.squeeze()):
        rec_movie_id = final_dataset.iloc[idx]["movieId"]

        title = movies[movies["movieId"] == rec_movie_id]["title"].values[0]
        avg_rating = ratings[ratings["movieId"] == rec_movie_id]["rating"].mean()

        recommendations.append({
            "Title": title,
            "Average Rating": round(avg_rating, 2),
            "Stars": "⭐" * int(round(avg_rating))
        })

    # Convert to DataFrame
    rec_df = pd.DataFrame(recommendations)

    # Remove the searched movie itself
    rec_df = rec_df[rec_df["Title"] != movie_list.iloc[0]["title"]]

    # Sort by rating
    rec_df = rec_df.sort_values(
        by="Average Rating", ascending=False
    ).reset_index(drop=True)

    return rec_df

# ============================================================
# 6. SAMPLE TEST (Optional)
# ============================================================
if __name__ == "__main__":
    print(get_top_rated_movies_list(5))
    print(get_recommendation("Vikram"))
