import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from logger import get_logger

logger = get_logger('data_preprocessing')

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        logger.info(f"Data loaded from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return None

def parse_floor(floor_str):
    if pd.isnull(floor_str):
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
        except ValueError:
            total_floors = np.nan
        return floor_num, total_floors

    elif floor_str.isdigit():
        return int(floor_str), np.nan
    elif floor_str == "Ground":
        return 0, np.nan
    else:
        return np.nan, np.nan


def preprocess_dataframe(df, median_floor=None, median_total_floors=None, fit=True):
    df[['Current_Floor', 'Total_Floors']] = df['Floor'].apply(lambda x: pd.Series(parse_floor(x)))
    if fit:
        median_floor = df['Current_Floor'].median()
        median_total_floors = df['Total_Floors'].median()
    df['Current_Floor'] = df['Current_Floor'].fillna(median_floor).astype(int)
    df['Total_Floors'] = df['Total_Floors'].fillna(median_total_floors).astype(int)

    df['Posted On'] = pd.to_datetime(df['Posted On'], errors='coerce')
    df['Month'] = df['Posted On'].dt.month
    df['Day'] = df['Posted On'].dt.day

    df = df[df['Point of Contact'] != 'Contact Builder']

    rent_upper = df['Rent'].quantile(0.99)
    size_upper = df['Size'].quantile(0.99)
    df = df[(df['Rent'] <= rent_upper) & (df['Size'] <= size_upper)]

    feature_cols = [
        'BHK', 'Size', 'Current_Floor', 'Total_Floors', 'Day', 'Month',
        'Area Locality', 'City', 'Furnishing Status', 'Tenant Preferred', 'Area Type', 'Bathroom'
    ]
    X = df[feature_cols]
    y = df['Rent']

    return X, y, median_floor, median_total_floors


def feature_engineering(X, top_areas=None, fit=True):
    if fit:
        top_areas = X['Area Locality'].value_counts().nlargest(20).index
    X = X.copy()
    X['Area_Locality_Top'] = X['Area Locality'].apply(lambda x: x if x in top_areas else 'Other')
    X = X.drop(columns=['Area Locality'])
    X['log_Size'] = np.log1p(X['Size'])
    X = X.drop(columns=['Size'])
    X['Bath_per_BHK'] = X['Bathroom'] / X['BHK']
    X['Size_per_BHK'] = np.log1p(X['log_Size'] / X['BHK'])
    X['Rel_Floor'] = X['Current_Floor'] / (X['Total_Floors'] + 1e-2)
    X['Is_Top_Floor'] = (X['Current_Floor'] == X['Total_Floors']).astype(int)
    X['Is_Ground_Floor'] = (X['Current_Floor'] == 0).astype(int)
    return X, top_areas


def split_and_prepare(df):
    X, y, median_floor, median_total_floors = preprocess_dataframe(df, fit=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, top_areas = feature_engineering(X_train, fit=True)
    X_test, _ = feature_engineering(X_test, top_areas, fit=False)
    return X_train, X_test, y_train, y_test, top_areas, median_floor, median_total_floors



