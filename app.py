import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image


# Load model

model = tf.keras.models.load_model(
    "cancer_detection_model.h5"
)


classes = [
    "Benign",
    "Malignant",
    "Normal"
]


st.title("🩺 AI Breast Cancer Detection System")

st.write(
    "Upload a breast ultrasound image for AI prediction."
)


uploaded_file = st.file_uploader(
    "Choose ultrasound image",
    type=["png","jpg","jpeg"]
)


if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )


    img = np.array(image)


    img = cv2.resize(
        img,
        (224,224)
    )


    img = img / 255.0


    img = np.expand_dims(
        img,
        axis=0
    )


    prediction = model.predict(img)


    result = classes[
        np.argmax(prediction)
    ]


    confidence = (
        np.max(prediction)
        *100
    )


    st.success(
        f"Prediction: {result}"
    )


    st.info(
        f"Confidence: {confidence:.2f}%"
    )