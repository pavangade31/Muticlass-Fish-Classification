import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

# ==== UPDATE THESE ====
MODEL_PATH = "mobilenet_transfer.h5"  # path to your saved model (relative or absolute)
IMG_SIZE = (150, 150)  # same image size used during training
CLASS_NAMES = [
    "animal fish", "animal fish bass", "fish sea_food black_sea_sprat",
    "fish sea_food gilt_head_bream", "fish sea_food hourse_mackerel",
    "fish sea_food red_mullet", "fish sea_food red_sea_bream",
    "fish sea_food sea_bass", "fish sea_food shrimp",
    "fish sea_food striped_red_mullet", "fish sea_food trout"
]
# ======================

# Load your trained model once when app starts
@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at: {MODEL_PATH}")
        st.stop()
    return load_model(MODEL_PATH)

model = load_trained_model()

# Streamlit UI
st.title("🐟 Fish Species Classification App")
st.write("Upload an image of a fish to classify it into one of the predefined species.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load and display the image
    image = load_img(uploaded_file, target_size=IMG_SIZE)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    preds = model.predict(img_array)
    pred_idx = np.argmax(preds)
    confidence = preds[0][pred_idx] * 100

    # Show results
    st.subheader("Prediction Results:")
    st.write(f"**Predicted Class:** {CLASS_NAMES[pred_idx]}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    # Show all class probabilities
    st.subheader("Confidence Scores for All Classes:")
    for i, class_name in enumerate(CLASS_NAMES):
        st.write(f"{class_name}: {preds[0][i]*100:.2f}%")