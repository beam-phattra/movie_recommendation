import os

from dotenv import load_dotenv

import uvicorn
from fastapi import FastAPI

from module.model_inference import load_model, recommend_movie_for_user
from module.dataset import get_dataset

load_dotenv("/configs/.env")

MODEL_PATH = os.getenv("MODEL_PATH", "")
DATASET_PATH = os.getenv("DATASET_PATH", "")

print(MODEL_PATH)
print(DATASET_PATH)


app = FastAPI()


@app.on_event("startup")
def load_clf():
    global model
    global movies_df
    global ratings_df

    model = load_model(MODEL_PATH)
    movies_df, ratings_df = get_dataset(DATASET_PATH)


@app.get("/")
async def index():
    return {"Message": "Movies Recommender API"}


@app.get("/recommendations")
async def predict(user_id: str, returnMetadata: bool = False):
    prediction = recommend_movie_for_user(model, user_id)
    movie_ids = [movie_id.decode("utf-8") for movie_id in prediction]
    result = {"items": [{"id": movie_id} for movie_id in movie_ids]}

    if not returnMetadata:
        return result

    items = result["items"]
    for item in items:
        movie_id = item["id"]
        title = movies_df[(movies_df["movieId"] == int(movie_id))]["title"].values[0]
        genre = movies_df[(movies_df["movieId"] == int(movie_id))]["genres"].values[0]
        item.update({"title": title})
        item.update({"genres": genre})
    return result


@app.get("/features")
async def get_features(user_id: str):
    history_list = (
        ratings_df[(ratings_df["userId"] == int(user_id))]["movieId"]
        .tail(4)
        .astype(int)
        .astype(str)
        .to_list()
    )
    result = {"features": [{"histories": history_list}]}
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
