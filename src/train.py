import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
import joblib
from data_preprocessing import split_and_prepare
import os
from logger import get_logger

logger = get_logger('train_model')

def train_model(df, model_path="../models/xgb_model.pkl"):
    X_train, X_test, y_train, y_test, top_areas, median_floor, median_total_floors = split_and_prepare(df)

    if not os.path.exists("../data/processed"):
        os.makedirs("../data/processed", exist_ok=True)
    X_train.to_csv("../data/processed/X_train.csv", index=False)
    X_test.to_csv("../data/processed/X_test.csv", index=False)
    y_train.to_csv("../data/processed/y_train.csv", index=False)
    y_test.to_csv("../data/processed/y_test.csv", index=False)
    logger.info("Processed CSV files saved in processed/")

    categorical_cols = ['Area_Locality_Top', 'City', 'Furnishing Status', 'Tenant Preferred', 'Area Type']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ],
        remainder='passthrough'
    )

    xgb_model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.1,
        subsample=1.0,
        random_state=42,
        verbosity=0
    )

    pipe = Pipeline([
        ('pre', preprocessor),
        ('model', xgb_model)
    ])

    y_train_log = np.log1p(y_train)
    pipe.fit(X_train, y_train_log)

    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

    joblib.dump({
        "model": pipe,
        "top_areas": top_areas,
        "median_floor": median_floor,
        "median_total_floors": median_total_floors,
        "categorical_cols": categorical_cols
    }, model_path)

    logger.info(f"Model saved to {model_path}")
    return pipe, X_test, y_test

if __name__ == "__main__":
    logger.info("Starting training...")
    df = pd.read_csv("../data/raw/House_Rent_Dataset.csv")
    train_model(df)
    logger.info("Training finished.")






