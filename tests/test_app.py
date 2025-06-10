import pytest
import numpy as np
from app.app import app, parse_floor, feature_engineering_input


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_homepage_loads(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Rent Price Prediction" in response.data

def test_prediction_valid(client):
    response = client.post("/", data={
        "BHK": 2,
        "Size": 1000,
        "Floor": "3 out of 10",
        "Bathroom": 2,
        "Area_Locality": "HSR Layout",
        "City": "Bangalore",
        "Furnishing_Status": "Semi-Furnished",
        "Tenant_Preferred": "Family",
        "Area_Type": "Super Area",
        "Posted_On": "2025-05-10",
    })
    assert response.status_code == 200
    assert b"Predicted Rent" in response.data

def test_prediction_invalid_floor(client):
    response = client.post("/", data={
        "BHK": 2,
        "Size": 1000,
        "Floor": "NotARealFloor",
        "Bathroom": 2,
        "Area_Locality": "HSR Layout",
        "City": "Bangalore",
        "Furnishing_Status": "Semi-Furnished",
        "Tenant_Preferred": "Family",
        "Area_Type": "Super Area",
        "Posted_On": "2025-05-10",
    })
    assert response.status_code == 200
    assert b"warning" in response.data.lower() or b"less accurate" in response.data.lower()

def test_parse_floor():
    assert parse_floor("5 out of 12") == (5, 12)
    assert parse_floor("Ground out of 5") == (0, 5)
    assert parse_floor("7") == (7, np.nan)
    assert parse_floor("Ground") == (0, np.nan)
    assert parse_floor("first floor") == (np.nan, np.nan)

def test_feature_engineering_input():
    row = {
        "BHK": 2,
        "Size": 1000,
        "Current_Floor": 2,
        "Total_Floors": 5,
        "Day": 10,
        "Month": 6,
        "Area Locality": "TestArea",
        "City": "Bangalore",
        "Furnishing Status": "Furnished",
        "Tenant Preferred": "Family",
        "Area Type": "Super Area",
        "Bathroom": 2,
    }
    fe_row = feature_engineering_input(row, top_areas=["TestArea"])
    assert "Area_Locality_Top" in fe_row
    assert "log_Size" in fe_row
    assert fe_row["Area_Locality_Top"] == "TestArea"