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

logger = get_logger("train_model")

MODEL_PATH = "models/xgb_model.pkl"
PROCESSED_DATA_PATH = "data/processed"
RAW_DATA_PATH = "data/raw/House_Rent_Dataset.csv"


class ModelTrainer:
    def __init__(
        self,
        model_path=MODEL_PATH,
        processed_data_path=PROCESSED_DATA_PATH,
        raw_data_path=RAW_DATA_PATH,
        logger_instance=None,
    ):
        self.model_path = model_path
        self.processed_data_path = processed_data_path
        self.raw_data_path = raw_data_path
        self.logger = logger_instance or logger

    def build_preprocessor(self, categorical_cols):
        return ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    categorical_cols,
                )
            ],
            remainder="passthrough",
        )

    def build_model_pipeline(self, categorical_cols):
        preprocessor = self.build_preprocessor(categorical_cols)

        xgb_model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.1,
            subsample=1.0,
            random_state=42,
            verbosity=0,
        )

        return Pipeline([("pre", preprocessor), ("model", xgb_model)])

    def train_model(self, df, model_path=None):
        target_model_path = model_path or self.model_path
        (
            X_train,
            X_test,
            y_train,
            y_test,
            top_areas,
            median_floor,
            median_total_floors,
        ) = split_and_prepare(df)

        if not os.path.exists(self.processed_data_path):
            os.makedirs(self.processed_data_path, exist_ok=True)
        X_train.to_csv(
            os.path.join(self.processed_data_path, "X_train.csv"), index=False
        )
        X_test.to_csv(os.path.join(self.processed_data_path, "X_test.csv"), index=False)
        y_train.to_csv(
            os.path.join(self.processed_data_path, "y_train.csv"), index=False
        )
        y_test.to_csv(os.path.join(self.processed_data_path, "y_test.csv"), index=False)
        self.logger.info("Processed CSV files saved in data/processed/")

        categorical_cols = [
            "Area_Locality_Top",
            "City",
            "Furnishing Status",
            "Tenant Preferred",
            "Area Type",
        ]
        pipe = self.build_model_pipeline(categorical_cols)

        y_train_log = np.log1p(y_train)
        pipe.fit(X_train, y_train_log)

        if not os.path.exists(target_model_path):
            os.makedirs(os.path.dirname(target_model_path), exist_ok=True)

        joblib.dump(
            {
                "model": pipe,
                "top_areas": top_areas,
                "median_floor": median_floor,
                "median_total_floors": median_total_floors,
                "categorical_cols": categorical_cols,
            },
            target_model_path,
        )

        self.logger.info(f"Model saved to {target_model_path}")
        return pipe, X_test, y_test


_model_trainer = ModelTrainer()


def train_model(df, model_path=MODEL_PATH):
    return _model_trainer.train_model(df, model_path)


if __name__ == "__main__":
    logger.info("Starting training...")
    df = pd.read_csv(RAW_DATA_PATH)
    _model_trainer.train_model(df)
    logger.info("Training finished.")
