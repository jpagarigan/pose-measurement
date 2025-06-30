# size_reco.py
import joblib
import json
import os
import numpy as np

# === File paths ===
MODEL_PATH = 'models/kmeans_15_model.joblib'
SCALER_PATH = 'models/scaler_15.joblib'
MAPPING_PATH = 'models/cluster_to_size_15.json'

# === Load model, scaler, and label map ===
kmeans_15 = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

with open(MAPPING_PATH, 'r') as f:
    cluster_to_size_15 = json.load(f)

# === Prediction function ===
def predict_size(shoulder, torso, waist):
    input_scaled = scaler.transform([[shoulder, torso, waist]])
    cluster = kmeans_15.predict(input_scaled)[0]
    return cluster_to_size_15[str(cluster)]
