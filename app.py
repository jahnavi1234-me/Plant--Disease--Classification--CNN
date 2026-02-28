import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

MODEL_PATH = "models/cnn_model.keras"
CLASS_NAMES_PATH = "models/class_names.json"
IMG_SIZE = (224, 224)

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