import os
import numpy as np
from PIL import Image
import cv2
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input

# Load the VGG19 model
base_model = VGG19(include_top=False, input_shape=(240, 240, 3),
                   weights=r'C:\Users\Vishnu\Documents\Advance_Brain_Tumor_Classification-main\vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')

x = base_model.output
x = Flatten()(x)
x = Dense(4608, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(1152, activation='relu')(x)
output = Dense(2, activation='softmax')(x)

model_03 = Model(inputs=base_model.input, outputs=output)
model_03.load_weights(r'C:\Users\Vishnu\Documents\Advance_Brain_Tumor_Classification-main\model_weights\vgg_unfrozen.h5')


app = Flask(__name__)

print('Model loaded. Check http://127.0.0.1:5000/')

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

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']
    if file.filename == '':
        return "No selected file"

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

if __name__ == '__main__':
    app.run(debug=True)
