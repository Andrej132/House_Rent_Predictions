import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn.model_selection import train_test_split
import os


def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Failed to load data: {e}")
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

def save_data(X_train, X_test, y_train, y_test, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)


def preprocess_and_save(file_path, output_dir):
    df = load_data(file_path)

    df[['Floor_num', 'Total_floors']] = df['Floor'].apply(lambda x: pd.Series(parse_floor(x)))
    df.drop(['Floor', 'Posted On', 'Point of Contact'], axis=1, inplace=True)

    df['Floor_num'] = df['Floor_num'].fillna(df['Floor_num'].median())
    df['Total_floors'] = df['Total_floors'].fillna(df['Total_floors'].median())

    rent_upper = df['Rent'].quantile(0.99)
    df = df[df['Rent'] < rent_upper]
    df['Rent_log'] = np.log1p(df['Rent'])

    target_encoder = ce.TargetEncoder(cols=['Area Locality'])
    df['Area Locality'] = target_encoder.fit_transform(df['Area Locality'], df['Rent_log'])

    categorical_cols = ['Area Type', 'City', 'Furnishing Status', 'Tenant Preferred']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype='int')

    X = df.drop(['Rent', 'Rent_log'], axis=1)
    y = df['Rent_log']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    save_data(X_train, X_test, y_train, y_test, output_dir)
    print(f"Data preprocessing complete. Files saved to {output_dir}.")



