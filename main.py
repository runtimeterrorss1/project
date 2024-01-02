import mlflow
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
    
    your_feature_columns = processed_data.columns.difference(['Reading', 'Timestamp'])

    feature_columns_list = list(your_feature_columns)
    
    your_feature_columns = pd.Index(feature_columns_list)

    print(your_feature_columns)
    # Create training and testing sets
    X_train = processed_data.loc[train_mask, your_feature_columns]
    y_train = processed_data.loc[train_mask, 'Reading']

    X_test = processed_data.loc[test_mask, your_feature_columns]
    y_test = processed_data.loc[test_mask, 'Reading']
    
    print("Doing Hypermeter Tuning...")
    param_grid = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.1, 0.01],
        'max_depth': [1, 3, 5, 7],
    }

    # Instantiate the XGBoost regressor
    xgb_model_hpt = XGBRegressor(objective='reg:squarederror')

    # Instantiate the GridSearchCV object
    grid_search = GridSearchCV(estimator=xgb_model_hpt, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)

    # Get the best parameters
    best_params = grid_search.best_params_

    print("Training the model on best parameters...")
    # Train an XGBoost model with the best parameters
    xgb_model_with_best_params = XGBRegressor(objective='reg:squarederror', **best_params)
    xgb_model_with_best_params.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = xgb_model_with_best_params.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    best_score = grid_search.best_score_

    print(f'Best Parameters: {best_params}')
    
    print(f'Root Mean Squared Error: {rmse}')
    print(f'Mean Absolute Error: {mae}')
    print(f'Grid Search Best Score: {best_score}')

    
    feature_importance_df = pd.DataFrame({'Feature': your_feature_columns, 'Importance': xgb_model_with_best_params.feature_importances_})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    
    print("Feature Importances:")
    print(feature_importance_df)
    
    
    
    # for feature in your_feature_columns:
    #     print(" Feature Name: ", feature, "   Importan")

    
    # print(processed_data.head(20))
    
    
    
    
    
    
    
    # pass


if __name__ == "__main__":
    run_main()

# mlflow.set_tracking_uri(uri="http://<host>:<port>")