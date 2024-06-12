import os
from dotenv import load_dotenv

from typing import Dict, Text

import numpy as np

import tensorflow as tf
import tensorflow_recommenders as tfrs

from app.model.data_preprocess import DataPreprocess

load_dotenv("configs/.env")

DATASET_PATH = os.getenv("DATASET_PATH")
MODEL_PATH = os.getenv("MODEL_PATH")

DATASET_SIZE = 100_000
TRAIN_SIZE = 80_000
SEED = 123

EMBEDDING_DIMENSION = 64

TRAIN_BATCH_SIZE = 8192
TEST_BATCH_SIZE = 4096
EPOCHS = 5


class UserModel(tf.keras.Model):
    def __init__(self, unique_user_ids):
        super().__init__()
        self.user_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.StringLookup(
                    vocabulary=unique_user_ids, mask_token=None
                ),
                tf.keras.layers.Embedding(
                    len(unique_user_ids) + 1, EMBEDDING_DIMENSION
                ),
            ]
        )

    def call(self, user_id):
        user_embedding = self.user_embedding(user_id)
        return user_embedding


class MovieModel(tf.keras.Model):
    def __init__(self, unique_movie_ids):
        super().__init__()
        self.movie_id_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.StringLookup(
                    vocabulary=unique_movie_ids, mask_token=None
                ),
                tf.keras.layers.Embedding(
                    len(unique_movie_ids) + 1, EMBEDDING_DIMENSION
                ),
            ]
        )
        self.genre_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(EMBEDDING_DIMENSION, activation="relu"),
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(EMBEDDING_DIMENSION, activation="relu"),
            ]
        )

    def call(self, movie_id, genres):
        movie_id_embedding = self.movie_id_embedding(movie_id)
        genre_embedding = self.genre_embedding(genres)
        return movie_id_embedding + genre_embedding


class TwoTowerModel(tfrs.Model):
    def __init__(
        self,
        user_model: tf.keras.Model,
        movie_model: tf.keras.Model,
        task: tf.keras.layers.Layer,
    ):
        super().__init__()
        self.user_model = user_model
        self.movie_model = movie_model
        self.task = task

    def compute_loss(
        self, features: Dict[Text, tf.Tensor], training=False
    ) -> tf.Tensor:
        user_embeddings = self.user_model(features["userId"])
        movie_embeddings = self.movie_model(features["movieId"], features["genres"])

        return self.task(user_embeddings, movie_embeddings)


def main():
    print("Creating training set...")
    ratings_tf, movies_tf = DataPreprocess(DATASET_PATH).run()
    shuffled = ratings_tf.shuffle(
        DATASET_SIZE, seed=SEED, reshuffle_each_iteration=False
    )

    train = shuffled.take(TRAIN_SIZE)
    test = shuffled.skip(TRAIN_SIZE).take(DATASET_SIZE - TRAIN_SIZE)

    user_ids = ratings_tf.batch(100_000).map(lambda x: x["userId"])
    movie_ids = movies_tf.batch(1_000).map(lambda x: x["movieId"])

    unique_user_ids = np.unique(np.concatenate(list(user_ids)))
    unique_movie_ids = np.unique(np.concatenate(list(movie_ids)))

    user_model = UserModel(unique_user_ids)
    movie_model = MovieModel(unique_movie_ids)

    metrics = tfrs.metrics.FactorizedTopK(
        candidates=movies_tf.batch(256).map(
            lambda x: (x["movieId"], movie_model(x["movieId"], x["genres"]))
        )
    )

    task = tfrs.tasks.Retrieval(metrics=metrics)

    model = TwoTowerModel(user_model, movie_model, task)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.05))

    cached_train = train.shuffle(DATASET_SIZE).batch(TRAIN_BATCH_SIZE).cache()
    cached_test = test.batch(TEST_BATCH_SIZE).cache()

    print("Training model...")
    model.fit(cached_train, epochs=EPOCHS)

    model.evaluate(cached_test, return_dict=True)

    # Create a model that takes in raw query features, and
    index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)

    path = os.path.join(MODEL_PATH, "model")

    # Save the index.
    tf.saved_model.save(index, path)
    print("Model Saved")


if __name__ == "__main__":
    main()
