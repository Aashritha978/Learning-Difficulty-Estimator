import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.set_page_config(
    page_title="Learning Content Difficulty Estimator",
    layout="centered"
)

@st.cache_resource
def load_cnn_model():
    model_path = "learning_content_difficulty_cnn.h5"

    if not os.path.exists(model_path):
        st.error("Model file not found! Please upload the model file.")
        return None

    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


model = load_cnn_model()

class_names = ["easy", "hard", "medium"]

st.title("📘 Learning Content Difficulty Estimator")

st.write(
    "Upload an image of study material to estimate its difficulty level."
)

if model is None:
    st.stop()

uploaded_file = st.file_uploader(
    "Upload an image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image")

        image = image.resize((128, 128))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]

        st.success(
            f"📊 Predicted Difficulty Level: **{predicted_class.capitalize()}**"
        )

    except Exception as e:
        st.error(f"Error processing image: {e}")
