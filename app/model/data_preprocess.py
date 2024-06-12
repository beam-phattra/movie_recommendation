import pandas as pd

import tensorflow as tf


class DataPreprocess:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def get_movies_df(self):
        movies_df = pd.read_csv(f"{self.dataset_path}/movies.csv", delimiter=",")
        movies_df["genres"] = movies_df["genres"].str.split("|", n=-1, expand=False)

        # Encoded genres
        unique_genres = list(
            set([genre for genres in movies_df["genres"] for genre in genres])
        )
        index = tf.keras.layers.StringLookup(vocabulary=unique_genres)
        encoder = tf.keras.layers.CategoryEncoding(
            num_tokens=index.vocabulary_size(), output_mode="multi_hot"
        )

        movies_df["genres_encoded"] = movies_df["genres"].apply(
            lambda x: encoder(index(x))
        )
        return movies_df

    def get_ratings_df(self):
        movies_df = self.get_movies_df()
        ratings_df = pd.read_csv(f"{self.dataset_path}/ratings.csv", delimiter=",")
        ratings_df = ratings_df.drop(["timestamp"], axis=1)
        ratings_df = ratings_df.dropna(subset=["userId"])
        ratings_df = ratings_df[ratings_df["rating"].between(0, 5)]
        ratings_df = ratings_df.merge(
            movies_df[["movieId", "title", "genres_encoded"]], on="movieId"
        )
        return ratings_df

    def run(self):
        ratings_df = self.get_ratings_df()
        movies_df = self.get_movies_df()

        ratings_tf = tf.data.Dataset.from_tensor_slices(
            {
                "userId": ratings_df["userId"].astype(str).tolist(),
                "movieId": ratings_df["movieId"].astype(str).tolist(),
                "genres": ratings_df["genres_encoded"].tolist(),
            }
        )
        movies_tf = tf.data.Dataset.from_tensor_slices(
            {
                "movieId": movies_df["movieId"].astype(str).tolist(),
                "genres": movies_df["genres_encoded"].tolist(),
            }
        )

        return ratings_tf, movies_tf
