import os
from dotenv import load_dotenv

import tensorflow as tf
import tensorflow_recommenders as tfrs

load_dotenv("configs/.env")
MODEL_PATH = os.getenv("MODEL_PATH")


def load_model(model_path):
    path = os.path.join(model_path, "model")
    model = tf.saved_model.load(path)
    return model


def recommend_movie_for_user(model, user_id: str, num_rec: int):
    _, titles = model([user_id])
    recommend_list = titles[0, :num_rec].numpy().tolist()
    return recommend_list
