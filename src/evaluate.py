import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import os
from logger import get_logger

logger = get_logger("evaluate_model")

MODEL_PATH = "models/xgb_model.pkl"
ARTIFACTS_DIR = "artifacts"
X_TEST_PATH = "data/processed/X_test.csv"
Y_TEST_PATH = "data/processed/y_test.csv"


class ModelEvaluator:
    def __init__(
        self,
        model_path=MODEL_PATH,
        artifacts_dir=ARTIFACTS_DIR,
        x_test_path=X_TEST_PATH,
        y_test_path=Y_TEST_PATH,
        logger_instance=None,
    ):
        self.model_path = model_path
        self.artifacts_dir = artifacts_dir
        self.x_test_path = x_test_path
        self.y_test_path = y_test_path
        self.logger = logger_instance or logger

    def evaluate_model(self, model, X_test, y_test):
        y_pred_log = model.predict(X_test)
        y_pred = np.expm1(y_pred_log)
        y_true = y_test.values

        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        self.logger.info(f"MSE: {mse:.2f}")
        self.logger.info(f"MAE: {mae:.2f}")
        self.logger.info(f"RMSE: {rmse:.2f}")
        self.logger.info(f"R^2: {r2:.2f}")

        return y_true, y_pred

    def plot_actual_vs_predicted(self, y_true, y_pred, save_path=None, n_points=1000):
        plt.figure(figsize=(8, 8))
        plt.scatter(y_true[:n_points], y_pred[:n_points], alpha=0.5)
        plt.plot(
            [min(y_true), max(y_true)], [min(y_true), max(y_true)], "r--", label="Ideal"
        )
        plt.xlabel("Actual Rent")
        plt.ylabel("Predicted Rent")
        plt.title("Actual vs Predicted Rent")
        plt.legend()
        plt.grid(True)
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            self.logger.info(f"Plot saved to {save_path}")
        plt.close()

    def plot_feature_importance(self, model, feature_names, save_path=None, top_n=20):
        importances = model.feature_importances_
        importance_df = sorted(zip(feature_names, importances), key=lambda x: -x[1])
        top = importance_df[:top_n]
        features, importances = zip(*top)
        plt.figure(figsize=(10, 6))
        plt.barh(features[::-1], importances[::-1], color="skyblue")
        plt.xlabel("Importance")
        plt.title(f"Top {top_n} Feature Importances")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            self.logger.info(f"Feature importance plot saved to {save_path}")
        plt.close()

    def run(self):
        self.logger.info("Loading model...")
        data = joblib.load(self.model_path)
        model = data["model"]

        X_test = pd.read_csv(self.x_test_path)
        y_test = pd.read_csv(self.y_test_path).squeeze()

        y_true, y_pred = self.evaluate_model(model, X_test, y_test)
        if not os.path.exists(self.artifacts_dir):
            os.makedirs(self.artifacts_dir, exist_ok=True)
        self.plot_actual_vs_predicted(
            y_true,
            y_pred,
            save_path=os.path.join(self.artifacts_dir, "actual_vs_predicted.png"),
        )

        ohe = model.named_steps["pre"].named_transformers_["cat"]
        categorical_cols = data["categorical_cols"]
        ohe_feature_names = ohe.get_feature_names_out(categorical_cols)
        num_feature_names = [
            col for col in X_test.columns if col not in categorical_cols
        ]
        all_feature_names = list(ohe_feature_names) + num_feature_names
        self.plot_feature_importance(
            model.named_steps["model"],
            all_feature_names,
            save_path=os.path.join(self.artifacts_dir, "feature_importance.png"),
        )


_model_evaluator = ModelEvaluator()


def evaluate_model(model, X_test, y_test):
    return _model_evaluator.evaluate_model(model, X_test, y_test)


def plot_actual_vs_predicted(y_true, y_pred, save_path=None, n_points=1000):
    return _model_evaluator.plot_actual_vs_predicted(
        y_true, y_pred, save_path, n_points
    )


def plot_feature_importance(model, feature_names, save_path=None, top_n=20):
    return _model_evaluator.plot_feature_importance(
        model, feature_names, save_path, top_n
    )


if __name__ == "__main__":
    _model_evaluator.run()
