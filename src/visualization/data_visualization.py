import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_distributions(df, columns):
    """Plot histograms for numerical columns."""
    for column in columns:
        plt.figure(figsize=(8, 6))
        df[column].hist(bins=30)
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.show()

def plot_correlation_matrix(df):
    """Plot a heatmap of the correlation matrix."""
    plt.figure(figsize=(10, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()

if __name__ == '__main__':
    # Example usage
    data_path = r"C:\Users\zezom\PycharmProjects\EduNova\Data\processed/cleaned_udemy_courses.csv"
    df = pd.read_csv(data_path)

    # Plot distributions for numerical columns
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    plot_distributions(df, numerical_columns)

    # Plot correlation matrix
    plot_correlation_matrix(df)
