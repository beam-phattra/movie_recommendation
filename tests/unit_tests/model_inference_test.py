import os

from app.module.model_inference import load_model, recommend_movie_for_user

MODEL_PATH = os.getenv("MODEL_PATH", "./models")

def test_recommend_movie_for_user():
    model = load_model(MODEL_PATH)
    user_id = 14
    recommend = recommend_movie_for_user(model, user_id=user_id)
    
    assert isinstance(recommend, list) and len(recommend) == 2