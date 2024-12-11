def encode_categorical(df, columns):
    """Encode categorical variables using one-hot encoding."""
    for column in columns:
        df = pd.get_dummies(df, columns=[column], prefix=column, drop_first=True)
    return df

if __name__ == "__main__":
    import pandas as pd
    data_path = r"C:\Users\zezom\PycharmProjects\EduNova\Data\processed/cleaned_udemy_courses.csv"
    df = pd.read_csv(data_path)
    df = encode_categorical(df, columns=['level', 'subject'])
    print(df.head())
