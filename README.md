#  Movie Recommendation System using KNN (Collaborative Filtering)

##  Project Overview
This project implements a **movie-to-movie recommendation system** using **collaborative filtering** based on user ratings.  
It applies the **K-Nearest Neighbors (KNN)** algorithm with **cosine similarity** to recommend movies similar to a given movie.

The system is designed for:
- Academic projects
- Final year submissions
- Internship and portfolio demonstrations

---

##  Features
- Item-based collaborative filtering
- KNN model with cosine similarity
- Sparse matrix optimization (CSR Matrix)
- Filters inactive users and unpopular movies
- Top-rated movie suggestions (post-2000)
- Ready for deployment (Streamlit / Hugging Face)

---

##  Recommendation Approach

### Technique Used
- **Collaborative Filtering**
- **Item-based recommendation**
- **Memory-based approach**

### Why KNN?
- Simple and interpretable
- Works well with sparse rating data
- No retraining required for new ratings
- Suitable for real-world recommendation systems

---

##  Dataset Description
The project uses datasets in **MovieLens format**.

### `movies.csv`
| Column | Description |
|------|------------|
| movieId | Unique movie identifier |
| title | Movie title with release year |
| genres | Movie genres |

