import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def preprocessing_data(data):
    # print(data)
    print("Checking for Null Values:")
    print(data.isnull().sum())
    
    data['Reading'].fillna(
    method='ffill', inplace=True)
    
    machine = LabelEncoder()
    sensor = LabelEncoder()

    data['Machine_ID'] = machine.fit_transform(data['Machine_ID'])
    data['Sensor_ID'] = sensor.fit_transform(data['Sensor_ID'])
    
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    data['dayofweek'] = data['Timestamp'].dt.dayofweek
    data['hour'] = data['Timestamp'].dt.hour
    data['weekofmonth'] = data['Timestamp'].apply(lambda x: (x.day-1)//7 + 1)

    data['Reading_Lag_1'] = data.groupby(['Machine_ID', 'Sensor_ID'])['Reading'].shift(1)
    data['Reading_Lag_3'] = data.groupby(['Machine_ID', 'Sensor_ID'])['Reading'].shift(3)
    
    data.dropna(inplace=True)
    
    return data