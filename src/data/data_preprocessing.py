import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Define input and output paths
RAW_DATA_DIR = r"C:\Users\zezom\PycharmProjects\EduNova\Data\raw"
PROCESSED_DATA_DIR = r"C:\Users\zezom\PycharmProjects\EduNova\Data\processed"

# Ensure output folder exists
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)


# Function for preprocessing dataset
def preprocess_dataset(file_name):
    # Define file paths
    raw_file_path = os.path.join(RAW_DATA_DIR, file_name)
    processed_file_path = os.path.join(PROCESSED_DATA_DIR, f"processed_{file_name}")

    # Load the raw dataset
    print(f"Loading raw Data from {raw_file_path}...")
    df = pd.read_csv(raw_file_path)

    # Drop missing values (example)
    print("Dropping missing values...")
    df = df.dropna()

    # Convert categorical variables to numeric (example)
    print("Encoding categorical variables...")
    df = pd.get_dummies(df)

    # Optional: Split Data into train/test for supervised tasks
    print("Splitting Data into train and test sets...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Save preprocessed Data
    print(f"Saving preprocessed Data to {processed_file_path}...")
    train_df.to_csv(processed_file_path.replace(".csv", "_train.csv"), index=False)
    test_df.to_csv(processed_file_path.replace(".csv", "_test.csv"), index=False)

    print("Preprocessing complete!")


# Example usage
if __name__ == "__main__":
    # Check if RAW_DATA_DIR exists
    if not os.path.exists(RAW_DATA_DIR):
        print(
            f"Error: The directory '{RAW_DATA_DIR}' does not exist. Please create it and place your raw CSV files there.")
    else:
        # List all files in the raw Data directory
        raw_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith(".csv")]

        # Check if there are any files in the directory
        if not raw_files:
            print(f"Error: No CSV files found in the directory '{RAW_DATA_DIR}'. Please add some files.")
        else:
            # Process each file
            for file in raw_files:
                preprocess_dataset(file)
