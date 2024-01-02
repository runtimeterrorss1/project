

def train_model(processed_data):
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
    
    
    
    
    pass