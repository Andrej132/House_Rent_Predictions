import joblib
from data_preprocessing import load_data, preprocess_and_save
import xgboost as xgb

def train():
    preprocess_and_save("../data/raw/House_Rent_Dataset.csv",'../data/processed')

    X_train = load_data("../data/processed/X_train.csv")
    y_train = load_data("../data/processed/y_train.csv")

    xgbr = xgb.XGBRegressor(colsample_bytree=1, learning_rate=0.1,max_depth=5,
                        n_estimators=200, subsample=0.8)
    xgbr.fit(X_train, y_train)
    joblib.dump(xgbr, '../models/xgboost_model.pkl')

if __name__ == "__main__":
    train()
    print("Training completed and model saved.")





