# app.py

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input

# Load model
model = tf.keras.models.load_model('chest_final_model.keras')

# Class names
class_names = ['COVID19', 'NORMAL', 'PNEUMONIA']

# Streamlit page config
st.set_page_config(page_title="Chest X-ray Classifier", layout="centered")

# Title
st.title('🩺 Chest X-ray Disease Detection')
st.markdown('''
Welcome to the Chest X-ray classification app.  
Upload a chest X-ray image, and the model will predict whether it is **COVID-19**, **Normal**, or **Pneumonia**.
''')

# Sidebar
st.sidebar.header('📤 Upload Image')
uploaded_file = st.sidebar.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')   # <-- force convert to RGB
    st.image(image, caption='🖼 Uploaded Chest X-ray', use_container_width=True)  # <-- fix deprecation warning

    st.write("")

    with st.spinner('⏳ Predicting... Please wait...'):
        # Preprocessing
        img = image.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)  # shape (1, 224, 224, 3)
        img_array = preprocess_input(img_array)

        # Prediction
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)

    # Result
    st.success(f"🎯 Prediction: **{predicted_class}**\n\n🔵 Confidence: **{confidence * 100:.2f}%**")

else:
    st.sidebar.info('👈 Upload a chest X-ray image to start prediction.')

# Footer
st.markdown("""
---
Made by Nowrin & Arup
""")
