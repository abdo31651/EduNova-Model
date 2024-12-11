import os
import pandas as pd
from .text_processing import preprocess_text
from .categorical_encoding import encode_categorical
from .scaling_normalization import scale_features


def feature_engineering(df):
    """Apply feature engineering steps to the dataset."""
    # Text Preprocessing
    print("Preprocessing text data...")
    df['course_title_processed'] = preprocess_text(df['course_title'])

    # Categorical Encoding
    print("Encoding categorical features...")
    df = encode_categorical(df, columns=['level', 'subject'])

    # Scaling Numerical Features
    print("Scaling numerical features...")
    df = scale_features(df, columns=['price', 'rating', 'num_subscribers', 'num_reviews'])

    return df


if __name__ == "__main__":
    # Define input and output paths
    data_path = r"C:\Users\zezom\PycharmProjects\EduNova\Data\processed\cleaned_udemy_courses.csv"
    output_path = r"C:\Users\zezom\PycharmProjects\EduNova\Data\processed\feature_engineered_data.csv"

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load data
    print("Loading data...")
    df = pd.read_csv(data_path)

    # Apply feature engineering
    print("Applying feature engineering...")
    df = feature_engineering(df)

    # Save the processed data
    print(f"Saving feature-engineered data to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Feature engineering complete!")
