import pandas as pd


def compute_statistics(df):
    """Compute and print basic statistics for a DataFrame."""
    print("Basic Statistics:\n")
    print(df.describe())  # Summary statistics for numerical columns
    print("\nMissing Values:\n")
    print(df.isnull().sum())  # Count missing values in each column
    print("\nCorrelation Matrix (Numerical Columns Only):\n")



if __name__ == '__main__':
    # Example usage
    data_path = r"C:\Users\zezom\PycharmProjects\EduNova\Data\processed/cleaned_udemy_courses.csv"  # Replace with the correct processed file path
    df = pd.read_csv(data_path)
    compute_statistics(df)