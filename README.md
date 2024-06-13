# Movie Recommendation

API for recommended movies for user using Tensorflow and FastAPI.

## How does it work

### Getting Started

We use google colab for model development and save model into `models` folder

### Prerequisites

Clone git repository:

```sh
git clone https://github.com/beam-phattra/movie_recommendation.git
```

Create virtual environment and activate environment:

```sh
python3.10 -m venv venv

source venv/bin/activate
```

### Installing

Install requirements run:

```sh
pip install -r requirements.txt
```

### Run tests

For unit test and integration test run:

```sh
python -m pytest tests/
```

### Deployment

A Dockerfile is provided to containerize the application. Build and run the Docker container using:

```sh
docker build -f Dockerfile -t movie-rec-api .

docker run -it -p 9070:9070 movie-rec-api
```

Open the application in any browser <http://0.0.0.0:9070/>

## How to feed input and get output

Open your browser and go to <http://0.0.0.0:9070/>

### GET /recommendations

parameter

- `user_id=20`

Open your browser and go to <http://0.0.0.0:9070/recommendations?user_id=20>

The API will return 2 movies id for user.

```json
{
    "items": [
        {"id": "2081"}, 
        {"id": "1064"}
    ]
}
```

parameter

- `user_id=20`
- `returnMetadata=True`

Open your browser and go to <http://0.0.0.0:9070/recommendations?user_id=20&returnMetadata=True>

The API will return 2 movies id for user with title and genres.

```json
{
    "items": [
        {
            "id": "2081",
            "title": "Little Mermaid, The (1989)",
            "genres": ["Animation", "Children", "Comedy", "Musical", "Romance"],
        },
        {
            "id": "1064",
            "title": "Aladdin and the King of Thieves (1996)",
            "genres": [
                "Animation",
                "Children",
                "Comedy",
                "Fantasy",
                "Musical",
                "Romance",
            ],
        },
    ]
}
```

### GET /features

parameter

- `user_id=20`

Open your browser and go to  <http://0.0.0.0:9070/features?user_id=20>

The API will return 4 most recent reviewed movie id by user.

```json
{
    "features":[
        {"histories":["6333","6345","6358","6365"]}
    ]
}
```

## How to improve in the future

- Enhanced Model Architecture
  - Implement deep learning models like Autoencoders to capture complex user-item interactions.
  - Use Embedding-based models where user and item embeddings are learned jointly.
  - Use grid search or randomized search for hyperparameter tuning to find the best model parameters.
  - Consider using Bayesian Optimization for a more efficient hyperparameter search.

- Enhanced Feature Engineering:
  - Use Natural Language Processing (NLP) techniques to analyze movie descriptions and reviews for additional features.
  - Apply techniques like TF-IDF, Word2Vec, or BERT embeddings to extract meaningful features from text data.
