import os

import tensorflow as tf


def load_model(model_path):
    path = os.path.join(model_path, "model")
    model = tf.saved_model.load(path)
    print(type(model))
    return model

def recommend_movie_for_user(model, user_id: str, num_rec: int=2):
    _, titles = model([str(float(user_id))])
    recommend_list = titles[0, :num_rec].numpy().tolist()
    return recommend_list