import mlflow
import pandas as pd
import numpy as np
from preprocessing_data import preprocessing_data



def load_data_csv():
    data = pd.read_csv("data/dummy_sensor_data.csv")
    
    return data


def run_main():
    print("Loading Data...")
    data = load_data_csv()
    print(data)
    
    print("Preprocessing Data Feature Engineering...")
    processed_data = preprocessing_data(data)
    
    print(processed_data.head(20))
    
    
    # pass


if __name__ == "__main__":
    run_main()

# mlflow.set_tracking_uri(uri="http://<host>:<port>")