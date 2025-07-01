import os
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
import requests
import re
import pandas as pd
import folium
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from flask import Flask, request, render_template, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from flask import Flask, render_template
from flask_cors import CORS
from config import Config
from models import db, bcrypt
from routes import auth

app = Flask(__name__)

app.config.from_object(Config)

# Initialize database and bcrypt
db.init_app(app)
bcrypt.init_app(app)
CORS(app)  # Enable CORS for frontend requests

app.register_blueprint(auth, url_prefix='/api')

GEOAPIFY_API_KEY = "1d913decd6e643f6baed14e74685fcb0"
UPLOAD_FOLDER = "static/uploads/"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained model
MODEL_PATH = "models/fracture_model.h5"

model = tf.keras.models.load_model(MODEL_PATH)

#symptom checker
login(token="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")
model_name = "manoj2423/medi"
subfolder = "model/checkpoint-14649"

tokenizer = AutoTokenizer.from_pretrained(model_name, subfolder=subfolder)
symptom_model = AutoModelForCausalLM.from_pretrained(model_name, subfolder=subfolder)



# ========== 1️⃣ LOAD BRAIN TUMOR MODEL ========== #
base_model = VGG19(include_top=False, input_shape=(240, 240, 3),
weights=r'C:\Users\Vishnu\Documents\dontuse\hydfull\Advance_Brain_Tumor_Classification-main\vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')

x = base_model.output
x = Flatten()(x)
x = Dense(4608, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(1152, activation='relu')(x)
output = Dense(2, activation='softmax')(x)

model_03 = Model(inputs=base_model.input, outputs=output)
model_03.load_weights(r'C:\Users\Vishnu\Documents\dontuse\hydfull\Advance_Brain_Tumor_Classification-main\vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')

print("✅ Brain Tumor Model Loaded")

# ========== 2️⃣ LOAD TF-IDF SPECIALTY MODEL ========== #
def train_tfidf():
    df = pd.read_csv(r"C:\Users\Vishnu\Documents\dontuse\hydfull\Advance_Brain_Tumor_Classification-main\BrainTumor Classification DL\health_issues_dataset.csv", delimiter=",")
    df.dropna(inplace=True)
    df["Health Issue"] = df["Health Issue"].apply(lambda text: re.sub(r"[^a-zA-Z0-9\s]", "", text.lower()))

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df["Health Issue"])
    y = df["Specialty"]

    model = LogisticRegression()
    model.fit(X, y)

    print("✅ TF-IDF Model Trained")
    return model, vectorizer

tfidf_model, tfidf_vectorizer = train_tfidf()

# ========== 3️⃣ FUNCTIONS FOR BRAIN TUMOR DETECTION ========== #
def get_class_name(class_no):
    return "Yes Brain Tumor" if class_no == 1 else "No Brain Tumor"

def get_result(img_path):
    try:
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError("Image not found or invalid image format.")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = image.resize((240, 240))
        image = np.array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)

        prediction = model_03.predict(image)
        class_index = np.argmax(prediction, axis=1)[0]
        return get_class_name(class_index)

    except Exception as e:
        return f"Error processing image: {e}"

# ========== 4️⃣ FUNCTIONS FOR HOSPITAL FINDER ========== #
def predict_specialty(health_issue):
    X_input = tfidf_vectorizer.transform([health_issue.lower()])
    return tfidf_model.predict(X_input)[0]

def get_nearest_hospital(latitude, longitude, specialty):
    url = f"https://api.geoapify.com/v2/places?categories=healthcare.hospital&filter=circle:{longitude},{latitude},20000&bias=proximity:{longitude},{latitude}&limit=1&apiKey={GEOAPIFY_API_KEY}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        if data["features"]:
            hospital = data["features"][0]
            name = hospital["properties"].get("name", "Unknown Hospital")
            address = hospital["properties"].get("formatted", "Unknown Address")
            lat = hospital["geometry"]["coordinates"][1]
            lon = hospital["geometry"]["coordinates"][0]
            return name, address, lat, lon
    return "Unknown Hospital", "Unknown Address", None, None

def create_map(hospital_name, hospital_lat, hospital_lon, user_lat, user_lon):
    map_path = r"C:\Users\Vishnu\Documents\dontuse\hydfull\Advance_Brain_Tumor_Classification-main\BrainTumor Classification DL\static\map.html"
    hospital_map = folium.Map(location=[user_lat, user_lon], zoom_start=12)
    folium.Marker([user_lat, user_lon], tooltip="Your Location", icon=folium.Icon(color="blue")).add_to(hospital_map)
    if hospital_lat and hospital_lon:
        folium.Marker([hospital_lat, hospital_lon], popup=hospital_name, icon=folium.Icon(color="red")).add_to(hospital_map)
    hospital_map.save(map_path)

# ========== 5️⃣ FLASK ROUTES ========== #
@app.route('/')
def home():
    return render_template('login.html')

@app.route('/home')
def home1():
    return render_template('home.html')
@app.route('/analyzer')
def analyzer():
    return render_template('analyzer.html')

@app.route('/mri_ct_scan')
def mri_ct_scan():
    return render_template('index.html')

@app.route('/fracture_detection', methods=["GET", "POST"])
def fracture_detection():
    MODEL_PATH = "models/fracture_model.h5"
    model = tf.keras.models.load_model(MODEL_PATH)
    if request.method == "POST":
        file = request.files["file"]
        if file:
            # Save file in `static/uploads/`
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            # Preprocess Image
            img = tf.keras.utils.load_img(filepath, target_size=(224, 224))
            img_array = tf.keras.utils.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            prediction = model.predict(img_array)[0][0]
            label = "Fractured" if prediction < 0.5 else "Not Fractured"

            return render_template("result.html", label=label, filename="uploads/" + file.filename)

    return render_template("fracture.html")


    

@app.route('/find_hospital')
def find_hospital():
    return render_template('indexhf.html')
@app.route("/tips")
def home2():
    return render_template("tips.html") 

@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return render_template('results.html')

    file = request.files['file']
    if file.filename == '':
        return render_template('results.html')

    try:
        basepath = os.path.dirname(__file__)
        upload_dir = os.path.join(basepath, 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, secure_filename(file.filename))
        file.save(file_path)

        result = get_result(file_path)
        return result

    except Exception as e:
        return f"Failed to process file: {e}"

@app.route("/hospital_results", methods=["POST"])
def hospital_results():
    health_issue = request.form["health_issue"]
    latitude = request.form["latitude"]
    longitude = request.form["longitude"]

    specialty = predict_specialty(health_issue)
    hospital_name, hospital_address, hospital_lat, hospital_lon = get_nearest_hospital(latitude, longitude, specialty)
    create_map(hospital_name, hospital_lat, hospital_lon, float(latitude), float(longitude))

    return redirect(url_for("results", specialty=specialty, name=hospital_name, address=hospital_address))

@app.route("/results")
def results():
    specialty = request.args.get("specialty", "Unknown")
    hospital_name = request.args.get("name", "Unknown Hospital")
    hospital_address = request.args.get("address", "Unknown Address")

    return render_template("results.html", specialty=specialty, name=hospital_name, address=hospital_address)

@app.route('/bmi')
def index():
    return render_template('bmi.html')

@app.route('/calculate_bmi', methods=['POST'])
def calculate_bmi():
    data = request.json
    weight = float(data.get('weight'))
    height = float(data.get('height'))
    unit = data.get('unit')

    if unit == "imperial":
        weight = weight * 0.453592  # Convert lbs to kg
        height = height * 2.54  # Convert inches to cm

    height = height / 100  # Convert cm to meters
    bmi = round(weight / (height * height), 2)

    category = ""
    if bmi < 18.5:
        category = "Underweight 🟡"
    elif bmi < 24.9:
        category = "Normal ✅"
    elif bmi < 29.9:
        category = "Overweight ⚠️"
    else:
        category = "Obese ❌"

    return jsonify({"bmi": bmi, "category": category})
@app.route("/symptom", methods=["GET", "POST"])
def predict():
    
    if request.method == "POST":
        try:
            symptoms = request.form.get("symptoms", "").strip()
            if not symptoms:
                return render_template("resultsy.html", info="No symptoms provided!")

            # Prepare input text
            input_text = f"Symptoms: {symptoms}\nPredict the disease and recommend treatment."
            inputs = tokenizer(input_text, return_tensors="pt")

            # Generate prediction
            with torch.no_grad():
                output = symptom_model.generate(**inputs, max_length=150)

            response_text = tokenizer.decode(output[0], skip_special_tokens=True)

            # Extract structured data
            medications = response_text.split("Medications:")[-1].split("\n")[0] if "Medications:" in response_text else "Not provided"
            diet = response_text.split("Diet:")[-1].split("\n")[0] if "Diet:" in response_text else "Not provided"
            precautions = response_text.split("Precautions:")[-1].split("\n")[0] if "Precautions:" in response_text else "Not provided"

            return render_template(
                "resultsy.html",
                symptoms=symptoms,
                disease_prediction=response_text.split("\n")[0],  # Extract first line as disease info
                medications=medications,
                diet=diet,
                precautions=precautions,
            )

        except Exception as e:
            return render_template("indexsy.html", info=f"Prediction failed: {e}")

    return render_template("indexsy.html")



if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)



# ========== 6️⃣ RUN FLASK APP ========== #
#if __name__ == "__main__":
    #print("🚀 Starting Flask App... Visit http://127.0.0.1:5000/")
    #app.run(debug=True)
