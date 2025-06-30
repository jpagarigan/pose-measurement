from flask import Flask, render_template, request
import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
import subprocess
from size_reco import predict_size
import csv
import datetime

# === Flask app ===
app = Flask(__name__)

# === Load pre-trained models ===
shoulder_model = joblib.load("models/model_shoulderpx_to_actualS.joblib")
torso_model = joblib.load("models/model_torsopx_to_actualT.joblib")
waist_model = joblib.load("models/model_waistpx_to_actualW.joblib")

# === MediaPipe Pose ===
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# === Helper Functions ===
def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def process_image(filepath, timestamp, label):
    image = cv2.imread(filepath)
    if image is None:
        return None, None

    # Resize image to width = 370
    h, w = image.shape[:2]
    new_w = 370
    new_h = int(h * new_w / w)
    image = cv2.resize(image, (new_w, new_h))
    annotated_image = image.copy()

    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return None, None

        landmarks = results.pose_landmarks.landmark
        h, w = image.shape[:2]

        mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        for idx, lm in enumerate(landmarks):
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.putText(annotated_image, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        def get_xy(index):
            return np.array([landmarks[index].x * w, landmarks[index].y * h])

        l_shoulder, r_shoulder = get_xy(11), get_xy(12)
        l_hip, r_hip = get_xy(23), get_xy(24)

        shoulderpx = calculate_distance(l_shoulder, r_shoulder)
        waistpx = calculate_distance(l_hip, r_hip)
        torsopx = (calculate_distance(l_shoulder, l_hip) + calculate_distance(r_shoulder, r_hip)) / 2

        actualS = shoulder_model.predict([[shoulderpx]])[0]
        actualT = torso_model.predict([[torsopx]])[0]
        actualW = waist_model.predict([[waistpx]])[0]

        predicted_size = predict_size(actualS, actualT, actualW)

        # Save to CSV
        csv_file = "predictions_log.csv"
        file_exists = os.path.isfile(csv_file)
        with open(csv_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["filename", "shoulderpx", "torsopx", "waistpx"])
            writer.writerow([os.path.basename(filepath), round(shoulderpx, 2), round(torsopx, 2), round(waistpx, 2)])

        # === Save annotated output to static/{timestamp}/{timestamp}_{label}_pose.jpg
        static_subfolder = os.path.join("static", timestamp)
        os.makedirs(static_subfolder, exist_ok=True)
        output_filename = f"{timestamp}_{label}_pose.jpg"
        output_path = os.path.join(static_subfolder, output_filename)
        cv2.imwrite(output_path, annotated_image)

        return {
            'shoulder': round(actualS, 2),
            'torso': round(actualT, 2),
            'waist': round(actualW, 2),
            'shoulderpx': round(shoulderpx, 2),
            'torsopx': round(torsopx, 2),
            'waistpx': round(waistpx, 2),
            'size': predicted_size,
            'output_image': os.path.join(timestamp, output_filename)
        }, output_path


# === Routes ===
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Step 1: Get images
    front_file = request.files['front']
    side_file = request.files['side']
    back_file = request.files['back']

    # Step 2: Create timestamped upload dir
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    upload_dir = os.path.join("uploads", timestamp)
    os.makedirs(upload_dir, exist_ok=True)

    # Step 3: Save uploaded images
    filepaths = {}
    for label, file in zip(["front", "side", "back"], [front_file, side_file, back_file]):
        filename = f"{timestamp}_{label}.jpg"
        full_path = os.path.join(upload_dir, filename)
        file.save(full_path)
        filepaths[label] = full_path

    # Step 4: Process each image after cropping
    results = {}
    images = {}

    for label in ["front", "side", "back"]:
        try:
            crop_result = subprocess.run(
                ["python", "auto_crop.py", filepaths[label]],
                capture_output=True,
                text=True,
                check=True
            )
            cropped_path = crop_result.stdout.strip()
            if not os.path.exists(cropped_path):
                raise FileNotFoundError("Cropped image not found.")

            result, image_path = process_image(cropped_path, timestamp, label)
            results[label] = result
            images[label] = result['output_image'].replace("\\", "/")

        except subprocess.CalledProcessError as e:
            results[label] = {"error": f"Cropping failed for {label}: {e.stderr}"}
            images[label] = None
        except Exception as e:
            results[label] = {"error": f"Processing failed for {label}: {str(e)}"}
            images[label] = None

    return render_template("display.html", result=results['front'], front_image=images['front'])


# === Run App ===
if __name__ == "__main__":
    app.run(debug=True)
