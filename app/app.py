from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "xgb_model.pkl")


class PredictionService:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.model_data = joblib.load(self.model_path)
        self.pipe = self.model_data["model"]
        self.top_areas = self.model_data["top_areas"]
        self.median_floor = self.model_data["median_floor"]
        self.median_total_floors = self.model_data["median_total_floors"]

    def parse_floor(self, floor_str):
        if not floor_str or pd.isnull(floor_str):
            return np.nan, np.nan
        floor_str = floor_str.strip()
        if "out of" in floor_str:
            part, total = floor_str.split("out of")
            part = part.strip()
            total = total.strip()
            if part == "Ground":
                floor_num = 0
            elif part == "Upper Basement":
                floor_num = -1
            elif part == "Lower Basement":
                floor_num = -2
            elif part.isdigit():
                floor_num = int(part)
            else:
                floor_num = np.nan
            try:
                total_floors = int(total)
            except Exception:
                total_floors = np.nan
            return floor_num, total_floors
        elif floor_str.isdigit():
            return int(floor_str), np.nan
        elif floor_str == "Ground":
            return 0, np.nan
        else:
            return np.nan, np.nan

    def feature_engineering_input(self, row, top_areas):
        row = row.copy()
        if row["Area Locality"] not in top_areas:
            row["Area_Locality_Top"] = "Other"
        else:
            row["Area_Locality_Top"] = row["Area Locality"]
        row.pop("Area Locality")
        row["log_Size"] = np.log1p(row["Size"])
        row.pop("Size")
        row["Bath_per_BHK"] = row["Bathroom"] / row["BHK"] if row["BHK"] else 0
        row["Size_per_BHK"] = (
            np.log1p(row["log_Size"] / row["BHK"]) if row["BHK"] else 0
        )
        row["Rel_Floor"] = (
            row["Current_Floor"] / (row["Total_Floors"] + 1e-2)
            if row["Total_Floors"]
            else 0
        )
        row["Is_Top_Floor"] = int(row["Current_Floor"] == row["Total_Floors"])
        row["Is_Ground_Floor"] = int(row["Current_Floor"] == 0)
        return row

    def predict_rent(self, form_data):
        warning = None
        user_data = {
            "BHK": int(form_data["BHK"]),
            "Size": float(form_data["Size"]),
            "Floor": form_data["Floor"],
            "Bathroom": int(form_data["Bathroom"]),
            "Area Locality": form_data["Area_Locality"],
            "City": form_data["City"],
            "Furnishing Status": form_data["Furnishing_Status"],
            "Tenant Preferred": form_data["Tenant_Preferred"],
            "Area Type": form_data["Area_Type"],
            "Posted On": pd.to_datetime(form_data.get("Posted_On", pd.Timestamp.now())),
        }
        current_floor, total_floors = self.parse_floor(user_data["Floor"])
        if np.isnan(current_floor) or np.isnan(total_floors):
            warning = (
                "Warning: The format of the Floor field is invalid. "
                "Typical values will be used instead. The predicted rent may be less accurate!"
            )
        user_data["Current_Floor"] = (
            int(current_floor)
            if not np.isnan(current_floor)
            else int(self.median_floor)
        )
        user_data["Total_Floors"] = (
            int(total_floors)
            if not np.isnan(total_floors)
            else int(self.median_total_floors)
        )
        user_data["Month"] = user_data["Posted On"].month
        user_data["Day"] = user_data["Posted On"].day
        feature_cols = [
            "BHK",
            "Size",
            "Current_Floor",
            "Total_Floors",
            "Day",
            "Month",
            "Area Locality",
            "City",
            "Furnishing Status",
            "Tenant Preferred",
            "Area Type",
            "Bathroom",
        ]
        row = {col: user_data[col] for col in feature_cols}
        row = self.feature_engineering_input(row, self.top_areas)
        model_cols = [
            "BHK",
            "Current_Floor",
            "Total_Floors",
            "Day",
            "Month",
            "City",
            "Furnishing Status",
            "Tenant Preferred",
            "Area Type",
            "Bathroom",
            "Area_Locality_Top",
            "log_Size",
            "Bath_per_BHK",
            "Size_per_BHK",
            "Rel_Floor",
            "Is_Top_Floor",
            "Is_Ground_Floor",
        ]
        row_df = pd.DataFrame([row])[model_cols]
        pred_log = self.pipe.predict(row_df)[0]
        pred_rent = np.expm1(pred_log)
        return f"{pred_rent:,.0f}INR", warning


class RentPredictionApp:
    def __init__(self, prediction_service=None):
        self.prediction_service = prediction_service or PredictionService()
        self.app = Flask(__name__)
        self._register_routes()

    def _register_routes(self):
        self.app.add_url_rule("/", view_func=self.index, methods=["GET", "POST"])

    def index(self):
        prediction = None
        error = None
        warning = None
        if request.method == "POST":
            try:
                prediction, warning = self.prediction_service.predict_rent(request.form)
            except Exception as e:
                error = f"Error in data processing: {str(e)}"
        return render_template(
            "index.html", prediction=prediction, error=error, warning=warning
        )


_prediction_service = PredictionService()
_rent_prediction_app = RentPredictionApp(_prediction_service)
app = _rent_prediction_app.app


def parse_floor(floor_str):
    return _prediction_service.parse_floor(floor_str)


def feature_engineering_input(row, top_areas):
    return _prediction_service.feature_engineering_input(row, top_areas)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
