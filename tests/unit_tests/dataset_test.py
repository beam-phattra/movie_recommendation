import os

import pandas as pd

from app.module.dataset import get_dataset

DATASET_PATH = os.getenv("DATASET_PATH", "./dataset")


def test_get_dataset():
    movies_df, ratings_df = get_dataset(DATASET_PATH)

    assert len(movies_df) > 0 and isinstance(movies_df, pd.DataFrame)
    assert len(ratings_df) > 0 and isinstance(ratings_df, pd.DataFrame)
