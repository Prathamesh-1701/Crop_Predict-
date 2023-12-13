from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "http://localhost:3000"}})

label_encoders = {
    'Region': joblib.load('./label_encoders/region_encoder.pkl'),
    'Month': joblib.load('./label_encoders/month_encoder.pkl'),
    'WeatherCondition': joblib.load('./label_encoders/weathercondition_encoder.pkl'),
    'Bulk Nutrient': joblib.load('./label_encoders/bulk nutrient_encoder.pkl'),
    'SeedQuality': joblib.load('./label_encoders/seedquality_encoder.pkl')
}

model = joblib.load('./model.pkl')

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = request.get_json().get('features', {})
        
        print("Received Features:", features)

        # Convert categorical features using label encoders
        for col, encoder in label_encoders.items():
            if col in features:
                # Handle previously unseen labels during prediction
                if features[col] not in encoder.classes_:
                    # Handle the unseen label based on your application's logic
                    # For simplicity, let's use the first class as a default value
                    features[col] = encoder.classes_[0]

                features[col] = encoder.transform([features[col]])[0]

        # Ensure that only required features are used for prediction
        required_features = ['Region', 'Month', 'WeatherCondition', 'Soil fertility', 'HungerIndex', 'MalnutritionRate', 'Bulk Nutrient', 'SeedQuality', 'Temperature']
        input_features = [features.get(col, 0) for col in required_features]

        # Perform the prediction using the loaded model
        result = model.predict([input_features])[0]

        return jsonify({'result': result.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
