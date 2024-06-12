import pandas as pd

def get_dataset(dataset_path):
    movies_df = pd.read_csv(f'{dataset_path}/movies.csv', delimiter=',')
    movies_df["genres"] = movies_df["genres"].str.split("|", n=-1, expand=False)
    
    ratings_df = pd.read_csv(f'{dataset_path}/ratings.csv', delimiter=',')
    return movies_df, ratings_df