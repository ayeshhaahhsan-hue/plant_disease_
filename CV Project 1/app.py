from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import pickle

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'my_model.keras')
ENCODER_PATH = os.path.join(BASE_DIR, 'encoder.pkl')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ================= MODEL + ENCODER =================
model = load_model(MODEL_PATH)
encoder = pickle.load(open(ENCODER_PATH, "rb"))

IMG_SIZE = (64, 64)

# ================= SMART GROUPED SYSTEM =================
# ✔ Common patterns per crop type
crop_nutrients = {
    "Apple": "Balanced NPK fertilizer + organic compost + potassium boost",
    "Tomato": "High potassium + calcium-rich fertilizer + nitrogen control",
    "Potato": "Low nitrogen + phosphorus-rich soil + organic manure",
    "Corn": "Nitrogen-rich fertilizer + zinc supplementation",
    "Grape": "Balanced NPK + magnesium support",
    "Default": "Balanced NPK fertilizer and healthy soil management"
}

crop_treatment = {
    "Apple": "Prune infected leaves, apply copper-based fungicide",
    "Tomato": "Remove infected parts, use fungicide spray regularly",
    "Potato": "Crop rotation + fungicide application + soil drainage",
    "Corn": "Use resistant seeds + proper field sanitation",
    "Grape": "Improve air circulation + fungicide treatment",
    "Default": "Consult agriculture expert for accurate treatment"
}

# ================= SMART FUNCTION =================
def get_crop(name):
    return name.split("___")[0] if "___" in name else "Default"


def get_info(disease):
    crop = get_crop(disease)

    return {
        "treatment": crop_treatment.get(crop, crop_treatment["Default"]),
        "nutrients": crop_nutrients.get(crop, crop_nutrients["Default"])
    }

# ================= HOME =================
@app.route('/')
def home():
    return render_template('index.html')


# ================= PREDICTION =================
@app.route('/predict', methods=['POST'])
def predict():

    file = request.files.get('file')

    if not file or file.filename == '':
        return render_template('index.html', error="Please upload image")

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # preprocess
    img = image.load_img(filepath, target_size=IMG_SIZE)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    # prediction
    prediction = model.predict(img)
    index = np.argmax(prediction[0])

    disease = encoder.inverse_transform([index])[0]

    info = get_info(disease)

    return render_template(
        "index.html",
        disease=disease,
        treatment=info["treatment"],
        nutrients=info["nutrients"],
        image_path=file.filename
    )


if __name__ == "__main__":
    app.run(debug=True)