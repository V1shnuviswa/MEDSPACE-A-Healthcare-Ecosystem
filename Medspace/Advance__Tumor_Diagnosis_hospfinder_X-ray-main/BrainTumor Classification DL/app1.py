from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import requests
import re
import folium
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

GEOAPIFY_API_KEY = "1d913decd6e643f6baed14e74685fcb0"

# ========== 1️⃣ TRAIN TF-IDF MODEL ==========
def train_tfidf():
    """Train TF-IDF + Logistic Regression model for specialty prediction (No saving)."""
    df =  pd.read_csv(r"C:\Users\Vishnu\Documents\Advance_Brain_Tumor_Classification-main\BrainTumor Classification DL\health_issues_dataset.csv", delimiter=",")
    df.dropna(inplace=True)

    df["Health Issue"] = df["Health Issue"].apply(lambda text: re.sub(r"[^a-zA-Z0-9\s]", "", text.lower()))

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df["Health Issue"])
    y = df["Specialty"]

    model = LogisticRegression()
    model.fit(X, y)

    print("✅ TF-IDF Model Trained at Runtime (No File Saving).")
    return model, vectorizer

tfidf_model, tfidf_vectorizer = train_tfidf()

# ========== 2️⃣ PREDICT SPECIALTY ==========
def predict_specialty(health_issue):
    """Predict medical specialty using trained TF-IDF model."""
    X_input = tfidf_vectorizer.transform([health_issue.lower()])
    return tfidf_model.predict(X_input)[0]

# ========== 3️⃣ FIND NEAREST HOSPITAL (GEOAPIFY) ==========
def get_nearest_hospital(latitude, longitude, specialty):
    """Find the nearest hospital using Geoapify Places API."""
    url = f"https://api.geoapify.com/v2/places?categories=healthcare.hospital&filter=circle:{longitude},{latitude},100000&bias=proximity:{longitude},{latitude}&limit=1&apiKey={GEOAPIFY_API_KEY}"
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

# ========== 4️⃣ GENERATE MAP ==========
def create_map(hospital_name, hospital_lat, hospital_lon, user_lat, user_lon):
    """Generate an interactive map with hospital location."""
    hospital_map = folium.Map(location=[user_lat, user_lon], zoom_start=12)
    
    # User location marker
    folium.Marker([user_lat, user_lon], tooltip="Your Location", icon=folium.Icon(color="blue")).add_to(hospital_map)
    
    if hospital_lat and hospital_lon:
        folium.Marker([hospital_lat, hospital_lon], popup=hospital_name, icon=folium.Icon(color="red")).add_to(hospital_map)

    hospital_map.save("static/map.html")

# ========== 5️⃣ FLASK ROUTES ==========
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        health_issue = request.form["health_issue"]
        latitude = request.form["latitude"]
        longitude = request.form["longitude"]

        specialty = predict_specialty(health_issue)
        hospital_name, hospital_address, hospital_lat, hospital_lon = get_nearest_hospital(latitude, longitude, specialty)
        create_map(hospital_name, hospital_lat, hospital_lon, float(latitude), float(longitude))

        return redirect(url_for("results", specialty=specialty, name=hospital_name, address=hospital_address))

    return render_template("indexhf.html")

@app.route("/results")
def results():
    specialty = request.args.get("specialty", "Unknown")
    hospital_name = request.args.get("name", "Unknown Hospital")
    hospital_address = request.args.get("address", "Unknown Address")

    return render_template("results.html", specialty=specialty, name=hospital_name, address=hospital_address)

# ========== 6️⃣ RUN FLASK APP ==========
if __name__ == "__main__":
    print("🚀 Starting Flask App... Visit http://127.0.0.1:5000/")
    app.run(debug=True)
