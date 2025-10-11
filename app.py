import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.applications.MobileNetV2(weights="imagenet")

# Streamlit UI
st.title("🧠 Deep Learning Image Classifier")
st.write("Upload an image and let the MobileNetV2 model classify it.")

# Component 1: File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Component 2: Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Component 3: Button to classify
    if st.button("Classify Image"):
        # Component 4: Progress bar
        progress = st.progress(0)
        for i in range(100):
            progress.progress(i + 1)

        # Predict
        preds = model.predict(img_array)
        decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=1)[0][0]
        label = decoded[1]
        confidence = decoded[2]

        # Component 5: Display result
        if confidence > 0.5:
            st.success(f"Prediction: {label} ({confidence:.2%} confidence)")
        else:
            st.error("Model is unsure about the prediction.")

