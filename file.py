# movie_recommendation_system.py

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------------
#  STEP 1: Create Sample Movie Ratings Data
# -----------------------------------------
# Rows = Users, Columns = Movies
data = {
    "User": ["A", "A", "A", "B", "B", "C", "C", "D", "D", "E", "E"],
    "Movie": [
        "Inception", "Avatar", "Titanic",
        "Inception", "Titanic",
        "Avatar", "Titanic",
        "Inception", "Avatar",
        "Titanic", "Avatar"
    ],
    "Rating": [5, 3, 4, 4, 5, 2, 5, 5, 4, 3, 4]
}

ratings_df = pd.DataFrame(data)
print(" Ratings Data:\n", ratings_df.head())

# -----------------------------------------
#  STEP 2: Create User–Item Matrix
# -----------------------------------------
user_movie_matrix = ratings_df.pivot_table(
    index="User",
    columns="Movie",
    values="Rating"
).fillna(0)

print("\n User–Movie Matrix:\n", user_movie_matrix)

# -----------------------------------------
#  STEP 3: Compute User Similarity (Collaborative Filtering)
# -----------------------------------------
user_similarity = cosine_similarity(user_movie_matrix)
user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_movie_matrix.index,
    columns=user_movie_matrix.index
)

print("\n User Similarity Matrix:\n", user_similarity_df.round(2))

# -----------------------------------------
#  STEP 4: Recommend Movies for a Given User
# -----------------------------------------
def get_recommendations(target_user, top_n=3):
    if target_user not in user_movie_matrix.index:
        print(" User not found!")
        return

    # Find similar users
    sim_users = user_similarity_df[target_user].sort_values(ascending=False)
    sim_users = sim_users.drop(target_user)  # remove self

    # Get the top similar users
    top_users = sim_users.head(2).index
    print(f"\n Similar users to {target_user}: {list(top_users)}")

    # Get movies rated by similar users but not by the target user
    target_ratings = user_movie_matrix.loc[target_user]
    unseen_movies = target_ratings[target_ratings == 0].index

    scores = {}
    for movie in unseen_movies:
        total_score = 0
        total_weight = 0
        for user in top_users:
            if user_movie_matrix.loc[user, movie] > 0:
                total_score += user_similarity_df.loc[target_user, user] * user_movie_matrix.loc[user, movie]
                total_weight += user_similarity_df.loc[target_user, user]
        if total_weight > 0:
            scores[movie] = total_score / total_weight

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    recommendations = [m for m, _ in sorted_scores[:top_n]]

    if recommendations:
        print(f"\n Recommended movies for {target_user}: {recommendations}")
    else:
        print(f"\n No new recommendations available for {target_user}.")

# -----------------------------------------
#  STEP 5: Try Recommending
# -----------------------------------------
get_recommendations("A")
get_recommendations("B")
get_recommendations("C")
