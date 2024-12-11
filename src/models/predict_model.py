import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD
import numpy as np

def prepare_dataset(data_path):
    # Load dataset
    df = pd.read_csv(data_path)

    # Simulate user interactions
    num_users = 500  # Example: 500 users
    df['user_id'] = np.random.randint(0, num_users, size=len(df))  # Simulate user IDs

    return df

def train_recommendation_model(df):
    # Create user-item interaction matrix
    interaction_matrix = coo_matrix(
        (df['rating'], (df['user_id'], df['course_id']))
    ).toarray()

    # Initialize Truncated SVD model
    svd = TruncatedSVD(n_components=50, random_state=42)

    # Train the model
    svd.fit(interaction_matrix)

    return svd, interaction_matrix

def predict_for_user(model, interaction_matrix, user_id, df):
    # Generate predictions for the given user
    user_interactions = interaction_matrix[user_id, :]
    predictions = model.inverse_transform(model.transform([user_interactions]))[0]

    # Get course recommendations
    df['predicted_rating'] = predictions[df['course_id']]
    recommendations = df[['course_id', 'course_title', 'predicted_rating']].sort_values(
        by='predicted_rating', ascending=False
    ).head(10)

    return recommendations

if __name__ == "__main__":
    data_path = r"C:\Users\zezom\PycharmProjects\EduNova\Data\processed/cleaned_udemy_courses.csv"

    # Prepare dataset
    df = prepare_dataset(data_path)

    # Train the model
    model, interaction_matrix = train_recommendation_model(df)

    # Specify a user ID to generate predictions for
    user_id = 10  # Replace with the desired user_id

    # Generate recommendations
    recommendations = predict_for_user(model, interaction_matrix, user_id, df)

    print("Top 10 Recommended Courses for User ID", user_id)
    print(recommendations)
