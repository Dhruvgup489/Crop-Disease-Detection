import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.title("Crop Disease Detection")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/crop_disease_model.keras")

model = load_model()

uploaded_file = st.file_uploader("Upload an image of the plant leaf", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    # Preprocessing & Prediction logic here