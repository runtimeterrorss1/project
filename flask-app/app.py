from flask import Flask, render_template, request, redirect, url_for
import pickle
import datetime
import pandas as pd

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
        
        df = pd.DataFrame({
            'Machine_ID': monitor_id,
            'Sensor_ID': sensor_id,
            'Timestamp': pd.to_datetime(timestamp),
        })
        df['dayofweek'] = df['Timestamp'].dt.dayofweek
        df['hour'] = df['Timestamp'].dt.hour
        df['weekofmonth'] = df['Timestamp'].dt.week // 4 + 1
        
        df['Reading_Lag_1'] = df.groupby(['Machine_ID', 'Sensor_ID'])['Reading'].shift(1)
        df['Reading_Lag_3'] = df.groupby(['Machine_ID', 'Sensor_ID'])['Reading'].shift(3)

        # Perform any necessary data preprocessing based on your model requirements
        # For example, convert timestamp to a format compatible with your model

        # Make predictions using your model
        prediction = model.predict(df)

        return render_template('results.html', prediction=prediction[0])

    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True, port=5002)
