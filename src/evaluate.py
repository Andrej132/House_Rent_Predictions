import joblib
import numpy as np

from data_preprocessing import load_data
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate():
    model = joblib.load('../models/xgboost_model.pkl')
    X_test = load_data("../data/processed/X_test.csv")
    y_test = load_data("../data/processed/y_test.csv")

    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_test)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"MSE: {mse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R^2: {r2:.2f}")

if __name__ == "__main__":
    evaluate()
    print("Evaluation completed.")