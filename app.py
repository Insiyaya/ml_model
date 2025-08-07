from flask import Flask, request, jsonify
import joblib
import numpy as np
import boto3
import os

AWS_ACCESS_KEY_ID = 'AKIA3UOYMTBEBHYZYKX4'
AWS_SECRET_ACCESS_KEY = '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'
S3_BUCKET_NAME = 'modellbuckett123'
MODEL_FILE_KEY = 'house_price_model.pkl'
MODEL_LOCAL_PATH = '/tmp/model.pkl'

def load_model():
    """Download model from S3 and load."""
    try:
        s3 = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )
        print(f"Downloading model from s3://{S3_BUCKET_NAME}/{MODEL_FILE_KEY}")
        s3.download_file(S3_BUCKET_NAME, MODEL_FILE_KEY, MODEL_LOCAL_PATH)
        return joblib.load(MODEL_LOCAL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

app = Flask(__name__)
model = load_model()

@app.route('/predict', methods=['GET'])
def predict():
    if not model:
        return jsonify({'error': 'Model is not loaded'}), 500
    try:
        data = request.get_json()
        features = [data['MedInc'], data['HouseAge'], data['AveRooms'], data['AveOccup']]
        prediction = model.predict(np.array([features]))
        return jsonify({'predicted_price': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 400