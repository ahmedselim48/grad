from flask import Flask, request, jsonify
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Endpoint for predicting
@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.get_json()

    # Convert the data to a numpy array
    new_data = np.array([
        data['Age'][0], 
        data['Sex'][0], 
        data['ChestPainType'][0],
        data['RestingBP'][0],
        data['Cholesterol'][0],
        data['FastingBloodSugar'][0],
        data['RestECG'][0],
        data['MaxHeartRate'][0],
        data['STSegmentSlope'][0]
    ]).reshape(1, -1)  # Reshape the data to be 2D

    # Scale the data
    scaled_data = scaler.transform(new_data)

    # Make prediction
    prediction = model.predict(scaled_data)

    # Convert prediction to a list and return as JSON response
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
