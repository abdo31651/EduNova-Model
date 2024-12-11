from sklearn.preprocessing import MinMaxScaler

def scale_features(df, columns):
    """Scale numerical features using MinMaxScaler."""
    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

if __name__ == "__main__":
    import pandas as pd
    data_path = r"C:\Users\zezom\PycharmProjects\EduNova\Data\processed/cleaned_udemy_courses.csv"
    df = pd.read_csv(data_path)
    df = scale_features(df, columns=['price', 'rating', 'num_subscribers', 'num_reviews'])
    print(df.head())
