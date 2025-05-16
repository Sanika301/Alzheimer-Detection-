from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import os
from keras.models import load_model, Model
from keras.applications.densenet import preprocess_input
from keras.preprocessing.image import img_to_array
import joblib

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Load Models ---
cnn_model = load_model("feature_extractor.h5")
feature_extractor = Model(inputs=cnn_model.input, outputs=cnn_model.layers[-2].output)
svm_model = joblib.load("svm_model.pkl")

# --- Class Names ---
class_names = ['AD', 'CN', 'MCI']

# --- Preprocessing Functions ---
def skull_stripping(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    brain = cv2.bitwise_and(image, image, mask=mask)
    return brain

def apply_median_filter(image):
    return cv2.medianBlur(image, 5)

def preprocess_single_image(img_path, target_size=(224, 224)):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Image not found: {img_path}")
    img = cv2.resize(img, target_size)
    img = skull_stripping(img)
    img = apply_median_filter(img)
    img = img_to_array(img)
    img = preprocess_input(img)
    return np.expand_dims(img, axis=0)

# --- Prediction Function ---
def predict_image(img_path):
    try:
        processed_img = preprocess_single_image(img_path)
        features = feature_extractor.predict(processed_img)
        probs = svm_model.predict_proba(features)[0]
        predicted_class_idx = np.argmax(probs)
        predicted_class = class_names[predicted_class_idx]
        confidence_scores = {label: round(float(score) * 100, 2) for label, score in zip(class_names, probs)}
        # confidence_scores = {label: round(float(score), 2) for label, score in zip(class_names, probs)}
        return predicted_class, confidence_scores
    except Exception as e:
        print(f"Prediction Error: {e}")
        return None, {}

# --- Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)

        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)

        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            prediction, confidence = predict_image(filepath)
            return render_template('index.html', prediction=prediction, confidence=confidence, img_path=filepath)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
