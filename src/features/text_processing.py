import re

def preprocess_text(column):
    """Clean and preprocess text data."""
    processed_column = column.str.lower()  # Convert to lowercase
    processed_column = processed_column.str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)  # Remove special characters
    processed_column = processed_column.str.strip()  # Remove leading/trailing spaces
    return processed_column

if __name__ == "__main__":
    import pandas as pd
    data_path = r"C:\Users\zezom\PycharmProjects\EduNova\Data\processed/cleaned_udemy_courses.csv"
    df = pd.read_csv(data_path)
    df['course_title_processed'] = preprocess_text(df['course_title'])
    print(df[['course_title', 'course_title_processed']].head())
