import os
import tensorflow as tf

# ------------------ Import Required Modules ------------------
Adam = tf.keras.optimizers.Adam
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
Sequential = tf.keras.models.Sequential
Conv2D = tf.keras.layers.Conv2D
MaxPooling2D = tf.keras.layers.MaxPooling2D
Flatten = tf.keras.layers.Flatten
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout

# ------------------ Dataset & Parameters ------------------

DATASET_DIR = r"C:\Users\Vishnu\Documents\Advance_Brain_Tumor_Classification-main\BrainTumor Classification DL\dataset\train"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
MODEL_PATH = "models/fracture_model.h5"

# Ensure model directory exists
os.makedirs("models", exist_ok=True)

# ✅ Data Augmentation & Preprocessing
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

# ✅ Load training & validation data
train_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training"
)

val_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation"
)

# ------------------ Build CNN Model ------------------

model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    
    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")  # Binary Classification (Fractured vs Normal)
])

# ✅ Compile Model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss="binary_crossentropy",
              metrics=["accuracy"])

# ------------------ Train Model ------------------

EPOCHS = 10

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# ✅ Save Model
model.save(MODEL_PATH)
print(f"✅ Model saved as {MODEL_PATH}")
