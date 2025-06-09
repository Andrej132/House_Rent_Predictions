# Rental Price Prediction Web App

## Project Overview

This project aims to build a machine learning application for predicting apartment rental prices based on property characteristics and location. The project includes data preprocessing, model training (with evaluation), testing, and a Flask web interface for user predictions. The entire solution is containerized using Docker and orchestrated with Docker Compose for easy setup and reproducibility.

---

## Dataset

The dataset consists of apartment rental listings from multiple Indian cities, with the following features:

- **BHK**: Number of Bedrooms, Hall, Kitchen.
- **Size**: Size of the apartment (in square feet).
- **Floor**: Floor information (e.g., "Ground out of 5", "3 out of 10").
- **Bathroom**: Number of bathrooms.
- **Area Locality**: Name of the locality/area.
- **City**: City where the property is located.
- **Furnishing Status**: Furnishing level (Furnished, Semi-Furnished, Unfurnished).
- **Tenant Preferred**: Preferred tenant type (Family, Bachelors, Bachelors/Family).
- **Area Type**: Type of area (Super Area, Built Area, Carpet Area).
- **Posted On**: Date the listing was posted.
- **Rent**: Target variable – monthly rent.
- **Point of Contact**: Who to contact for the listing (filtered during preprocessing).

---

## Technologies Used

- **Python 3.12**
- **Flask** – Web API for user interaction
- **scikit-learn** – Data preprocessing and pipeline
- **XGBoost** – Machine learning regression model
- **Pandas, NumPy** – Data manipulation
- **matplotlib** – Evaluation plots
- **Pytest** – Unit and integration testing
- **Docker, Docker Compose** – Containerization
- **Joblib** – Model serialization
- **Logging** – For unified logging across training, evaluation, and serving

---

## Project Structure

```
project-root/
│
├── app/
│   ├── app.py
│   ├── static/
│   │   ├── style.css
│   └── templates/
│       └── index.html
│
├── src/
│   ├── train.py
│   ├── evaluate.py
│   ├── data_preprocessing.py
│   └── logger.py
│
├── tests/
│   └── test_app.py
│
├── data/
│   ├── raw/
│   │   └── House_Rent_Dataset.csv
│   └── processed/
│        └──   # (generated processed data)
│
├── models/
│   └──   # (saved model files)
│
├── artifacts/
│   └──   # (generated plots)
│
├── notebooks/
│   └── EDA_&_Modeling.ipynb
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## How to Run the Project

### 1. **Prerequisites**

- [Docker](https://www.docker.com/get-started) and [Docker Compose](https://docs.docker.com/compose/) installed

### 2. **Build and Run All Services**

From the project root, run:

```bash
docker compose up --build
```

This will start the following services (in the correct order):

- **train**: Trains the XGBoost model and saves processed data and model artifacts.
- **evaluate**: Loads the trained model, evaluates it, and generates plots.
- **test**: Runs all unit/integration tests using pytest.
- **web**: Launches the Flask web app for user predictions.

The web app will be available at: [http://localhost:5000](http://localhost:5000)

### 3. **Stopping the Project**

To stop all containers:

```bash
docker compose down
```

---

## Usage

- Open your browser and go to [http://localhost:5000](http://localhost:5000)
- Fill in the apartment details in the web form and submit to get a rent prediction.
- If some fields (like Floor) are incorrectly formatted, the app will warn you and use default values.
