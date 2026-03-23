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

- [Docker](https://www.docker.com/get-started) and Docker Compose installed
- Run all commands from the project root

### 2. **Build the Image Once**

Build the shared image before running any stage of the pipeline:

```bash
docker compose build
```

All services use the same image: training, evaluation, testing, and the web app.

### 3. **Recommended End-to-End Workflow**

If you want to go through the full pipeline in order, use these commands:

#### Step 1: Train the Model

```bash
docker compose up train
```

What this does:

- Loads the raw dataset from `data/raw/`
- Preprocesses the data
- Saves processed datasets to `data/processed/`
- Saves the trained model to `models/`

#### Step 2: Evaluate the Model

```bash
docker compose --profile pipeline up evaluate
```

What this does:

- Uses the trained model from `models/`
- Reads test data from `data/processed/`
- Generates evaluation plots in `artifacts/`

#### Step 3: Run the Tests

```bash
docker compose --profile test up test
```

What this does:

- Runs the automated test suite with `pytest`
- Verifies the Flask app and prediction flow against the trained model

#### Step 4: Start the Web Application

```bash
docker compose up web
```

The application will be available at [http://localhost:5000](http://localhost:5000).

To stop the web container:

```bash
docker compose down
```

### 4. **Quick Command Reference**

Use these when you only need one stage:

```bash
docker compose build
docker compose up train
docker compose --profile pipeline up evaluate
docker compose --profile test up test
docker compose up web
docker compose down
```

### 5. **Outputs Created by Each Stage**

- Training creates processed CSV files in `data/processed/`
- Training creates the saved model in `models/xgb_model.pkl`
- Evaluation creates plots in `artifacts/`
- Web uses `models/xgb_model.pkl` to serve predictions

### 6. **One-Command Options**

If you only want to run the app and let Compose handle the required training step automatically:

```bash
docker compose up --build
```

This starts:

- `train`
- `web`

If you want to run training and evaluation together:

```bash
docker compose --profile pipeline up --build
```

This starts:

- `train`
- `evaluate`

If you want to run training and tests together:

```bash
docker compose --profile test up --build
```

This starts:

- `train`
- `test`

### 7. **Reset Generated Outputs**

If you want a clean run of the pipeline, remove generated outputs first:

```bash
docker compose down
docker compose rm -f
```

Optionally delete generated local folders if you want to retrain from scratch.

PowerShell:

```powershell
Remove-Item -Recurse -Force models, artifacts, data/processed
```

Bash:

```bash
rm -rf models artifacts data/processed
```

---

## Usage

- Open your browser and go to [http://localhost:5000](http://localhost:5000)
- Fill in the apartment details in the web form and submit to get a rent prediction.
- If some fields (like Floor) are incorrectly formatted, the app will warn you and use default values.
