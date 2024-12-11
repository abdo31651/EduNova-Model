import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np

def prepare_dataset(data_path):
    # Load dataset
    df = pd.read_csv(data_path)

    # Simulate user interactions
    num_users = 500  # Example: 500 users
    df['user_id'] = np.random.randint(0, num_users, size=len(df))  # Simulate user IDs

    # Ensure required columns exist
    if 'course_id' not in df.columns:
        df['course_id'] = df['course_id']  # Use the existing course_id column
    if 'rating' not in df.columns:
        df['rating'] = df['rating'].fillna(0.5)  # Default rating value if missing

    return df

def train_recommendation_model(data_path):
    # Prepare dataset
    df = prepare_dataset(data_path)

    # Create user-item interaction matrix
    interaction_matrix = coo_matrix(
        (df['rating'], (df['user_id'], df['course_id']))
    ).toarray()

    # Split into training and test sets
    train_matrix, test_matrix = train_test_split(interaction_matrix, test_size=0.2, random_state=42)

    # Initialize Truncated SVD model
    svd = TruncatedSVD(n_components=50, random_state=42)

    # Train the model on the training matrix
    svd.fit(train_matrix)

    # Generate predictions
    predicted_matrix = svd.inverse_transform(svd.transform(test_matrix))

    return svd, train_matrix, test_matrix, predicted_matrix

def evaluate_model(test_matrix, predicted_matrix):
    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(test_matrix[test_matrix > 0], predicted_matrix[test_matrix > 0])
    return mse

if __name__ == "__main__":
    data_path = r"C:\Users\zezom\PycharmProjects\EduNova\Data\processed/cleaned_udemy_courses.csv"

    # Train the model
    model, train_matrix, test_matrix, predicted_matrix = train_recommendation_model(data_path)

    # Evaluate the model
    mse = evaluate_model(test_matrix, predicted_matrix)
    print(f"Model trained successfully. Mean Squared Error (MSE): {mse:.4f}")
