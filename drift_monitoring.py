import pandas as pd
import numpy as np
from preprocessing_data import preprocessing_data
from train_model import train_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
import subprocess

import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature
import pickle

def load_pretrained_model(model_file_path):
    with open(model_file_path, 'rb') as model_file:
        model = pickle.load(model_file)
    return model

def func_bestrun(Expid, nameMetric):
    client = mlflow.tracking.MlflowClient(tracking_uri="http://127.0.0.1:5001")
    best_runs = client.search_runs(
        experiment_ids=[Expid],
        order_by=[f"metrics.{nameMetric} ASC"],
        max_results=1,
    )

    if not best_runs or pd.isna(best_runs[0]['metrics.' + nameMetric]):
        raise ValueError(f"No runs found for experiment {Expid} with metric {nameMetric}")

    best_run = best_runs[0]
    return best_run

def load_data_csv():
    data = pd.read_csv("data/dummy_sensor_data.csv")

    return data


def run_main():
    print("Loading Data...")
    data = load_data_csv()
    print(data)

    print("Preprocessing Data Feature Engineering...")
    processed_data = preprocessing_data(data)

    print("Data Splitting based on date:")
    last_date = processed_data['Timestamp'].max()

    # Set the split date as 1 day before the last date
    split_date = last_date - pd.DateOffset(hours=24)

    # Split the data into training and testing sets based on the split date
    train_mask = processed_data['Timestamp'] < split_date
    test_mask = ~train_mask

    your_feature_columns = processed_data.columns.difference(
        ['Reading', 'Timestamp'])

    feature_columns_list = list(your_feature_columns)

    your_feature_columns = pd.Index(feature_columns_list)

    print(your_feature_columns)
    # Create training and testing sets
    X_train = processed_data.loc[train_mask, your_feature_columns]
    y_train = processed_data.loc[train_mask, 'Reading']

    X_test = processed_data.loc[test_mask, your_feature_columns]
    y_test = processed_data.loc[test_mask, 'Reading']

    print(X_train)
    
    model_file_path = 'xgboost_model.pkl'
    loaded_model = load_pretrained_model(model_file_path)

    # Make predictions on the test set
    predictions = loaded_model.predict(X_test)

    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_test, predictions)
    threshold = 100

    print("MAE:", mae)

    # Trigger retraining if MAE exceeds the threshold
    if mae > threshold:
        print("Retraining triggered!")
        subprocess.call(['python', 'main.py'])
    
    


if __name__ == "__main__":
    run_main()
