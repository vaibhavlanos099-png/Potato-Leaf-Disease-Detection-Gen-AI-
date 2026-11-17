import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

# Page configuration
st.set_page_config(page_title="Potato CNN", page_icon="🥔", layout="wide")

ROOT = Path(__file__).resolve().parent
IMG_SIZE = (128, 128)

# Load model and class names
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(ROOT / "potato_cnn_model.keras")

@st.cache_data
def load_classes():
    return (ROOT / "classification_report.txt").read_text().splitlines()

# Sidebar
st.sidebar.title("🥔 Potato AI")
page = st.sidebar.radio("Navigation", ["Home", "About"])

if page == "Home":
    st.title("🥔 Potato Leaf Classification")
    st.write("Upload an image to predict its class.")
    st.divider()

    uploaded_file = st.file_uploader("Upload JPG Image", type=["jpg", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Uploaded Image")
            st.image(image, width="stretch")

        with col2:
            st.subheader("Prediction")

            if st.button("Predict Image", type="primary"):
                try:
                    model = load_model()
                    classes = load_classes()

                    image = image.resize(IMG_SIZE)
                    x = np.expand_dims(np.array(image, dtype=np.float32), axis=0)

                    scores = model.predict(x, verbose=0)[0]
                    best = np.argmax(scores)

                    st.success("Prediction Completed!")
                    st.metric("Predicted Class", classes[best].replace("potato__", ""))
                    st.metric("Confidence", f"{scores[best]*100:.2f}%")

                    st.subheader("Class Probabilities")
                    for name, score in zip(classes, scores):
                        st.write(f"{name}: {score*100:.2f}%")
                        st.progress(float(score))

                except Exception as e:
                    st.error(f"Prediction Error: {e}")

else:
    st.title("About Project")
    st.write("Potato Leaf Classification using CNN.")
    st.write("Framework: TensorFlow / Keras")
    st.write("Frontend: Streamlit")
    st.write("Image Size: 128 × 128")
    st.write("Output Classes: 3")