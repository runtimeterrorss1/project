from flask import Flask, render_template, request, redirect, url_for
import pickle
import datetime
import pandas as pd
import random


app = Flask(__name__)

# Load your AI model using pickle (replace 'your_model.pkl' with your actual model file)
with open('xgboost_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        timestamp = request.form['timestamp']
        monitor_id = int(request.form['monitor_id'])
        sensor_id = int(request.form['sensor_id'])

        print(timestamp)

        df = pd.DataFrame({
            'Machine_ID': [monitor_id],
            'Reading_Lag_1': [random.randrange(50, 201)],
            'Reading_Lag_3': [random.randrange(50, 201)],
            'Sensor_ID': [sensor_id],
            'Timestamp': [pd.to_datetime(timestamp)],
        })
        df['dayofweek'] = df['Timestamp'].dt.dayofweek
        df['hour'] = df['Timestamp'].dt.hour    
        df['weekofmonth'] = df['Timestamp'].apply(lambda x: (x.day-1)//7 + 1)

        # Generate random values for 'Reading_Lag_1' and 'Reading_Lag_3'
        df = df.drop('Timestamp', axis=1)

        # Perform any necessary data preprocessing based on your model requirements
        # For example, convert timestamp to a format compatible with your model

        # Make predictions using your model
        print("predictions: ")
        print(df)
        prediction = model.predict(df)
        print("predictions:", prediction)

        # return render_template('results.html', prediction="Success")
        return render_template('results.html', prediction=prediction[0])

    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True, port=5002)
