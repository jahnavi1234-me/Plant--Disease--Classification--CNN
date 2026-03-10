import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import requests
import os

# Ensure models folder exists
MODEL_DIR = "/tmp/models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Paths for model and class names
MODEL_PATH = os.path.join(MODEL_DIR, "cnn_model.keras")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")
IMG_SIZE = (224, 224)

# URLs to download your files (replace with your real links)
MODEL_URL = "https://your-cloud-link/cnn_model.keras"
CLASS_NAMES_URL = "https://your-cloud-link/class_names.json"

# Download model if missing
if not os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "wb") as f:
        f.write(requests.get(MODEL_URL).content)

# Download class names if missing
if not os.path.exists(CLASS_NAMES_PATH):
    with open(CLASS_NAMES_PATH, "wb") as f:
        f.write(requests.get(CLASS_NAMES_URL).content)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

st.title("🌿 Plant Disease Detection")

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    st.success(f"Predicted Disease: {class_names[class_index]}")
    st.info(f"Confidence: {confidence:.2f}")
