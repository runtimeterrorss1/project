import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
from flask import Flask, jsonify

app = Flask(__name__)

Id_exp = "0"  # Replace with your experiment ID
name_Metric = "mse"  # Replace with your metric name

def load_best_model():
    client = mlflow.tracking.MlflowClient()
    best_run = client.search_runs(
        experiment_ids=[Id_exp],
        order_by=[f"metrics.{name_Metric} DESC"]
    )
    if not best_run:
        raise ValueError("No runs found for the experiment")
    
    best_run_df = pd.DataFrame(best_run)
    
    # Print the columns to identify the correct column name for run ID
    print("Columns:", best_run_df.columns)
    
    # Replace 'run_id' with the correct column name for run ID
    run_id = best_run_df.iloc[0]['<correct_column_name_for_run_id>']
    model_path = f"/mlruns/0/{run_id}/artifacts/best_model"
    best_model = mlflow.sklearn.load_model(model_path)
    return best_model


data = {
    "Timestamp": ["2023-01-01 00:00:00"] * 15,
    "Machine_ID": ["Machine_1", "Machine_2", "Machine_3", "Machine_4", "Machine_5"],
    "Sensor_ID": ["Sensor_1", "Sensor_2", "Sensor_3"] * 5,
    "Reading": [109.93, 110.12, 108.56, 111.34, 109.78,
                110.45, 109.96, 107.89, 112.00, 108.67]
}

best_model = load_best_model()

@app.route('/predict', methods=['GET'])
def predict():
    try:
        df = pd.DataFrame(data)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['Hours'] = df['Timestamp'].dt.hour
        df['Date'] = df['Timestamp'].dt.day
        df['Month'] = df['Timestamp'].dt.month
        X = df[['Hours', 'Date', 'Month', 'Machine_ID', 'Sensor_ID']]
        #standard Scalar for scaling -> numeric
        #one hot encoder for encoding -> categorical
        #numeric cols
        cols1= ['Hours', 'Date', 'Month']
        #categorical cols
        cols2= ['Machine_ID', 'Sensor_ID']

        scaled = StandardScaler()
        encoded= OneHotEncoder(handle_unknown='ignore')

        preprocessing = ColumnTransformer(
            transformers=[
                ('num', scaled, cols1),
                ('cat', encoded, cols2)
            ])
        # X_transformed = preprocessing.fit_transform(X)
        X_df = pd.DataFrame(X, columns=['Hours', 'Date', 'Month', 'Machine_ID', 'Sensor_ID'])
        X_transformed = preprocessing.fit_transform(X_df)

        predictions = best_model.predict(X_transformed)
        print("Prediction:", predictions)
        # predictions as JSON returnn
        return jsonify({'predictions': predictions.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')