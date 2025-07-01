import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, url_for

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads/"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained model
MODEL_PATH = "models/fracture_model.h5"

model = tf.keras.models.load_model(MODEL_PATH)

@app.route("/fracture_detection", methods=["GET", "POST"])
def upload():
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

if __name__ == "__main__":
    app.run(debug=True)
