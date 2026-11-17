
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

# Page Configuration
st.set_page_config(
    page_title="Potato Leaf Disease Detection",
    page_icon="🥔",
    layout="centered"
)

ROOT = Path(__file__).resolve().parent
IMG_SIZE = (128, 128)

# Load Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        ROOT / "potato_cnn_model.keras"
    )

# Load Classes
@st.cache_data
def load_classes():
    return (ROOT / "classification_report.txt").read_text().splitlines()


# Main UI
st.title("🥔 Potato Leaf Disease Detection")

st.write(
    "Upload a potato leaf image to identify whether "
    "the leaf is healthy or affected by disease."
)

st.divider()

# Image Upload
uploaded_file = st.file_uploader(
    "Choose Potato Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Potato Leaf",
        width=300
    )

    if st.button(
        "🔍 Predict Disease",
        type="primary",
        use_container_width=True
    ):

        try:
            model = load_model()
            classes = load_classes()

            img = image.resize(IMG_SIZE)

            x = np.expand_dims(
                np.array(img, dtype=np.float32),
                axis=0
            )

            scores = model.predict(x, verbose=0)[0]

            best = np.argmax(scores)

            predicted_class = classes[best].replace(
                "potato__", ""
            )

            confidence = scores[best] * 100

            st.divider()

            st.success("Prediction Completed Successfully!")

            st.subheader("Prediction Result")

            st.write(f"**Disease:** {predicted_class}")

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

            st.divider()

            st.subheader("Class Probabilities")

            for name, score in zip(classes, scores):

                class_name = name.replace("potato__", "")

                st.write(
                    f"{class_name}: {score * 100:.2f}%"
                )

                st.progress(float(score))

        except Exception as e:

            st.error(f"Prediction Error: {e}")

else:
    st.info("Please upload a potato leaf image to continue.")
