"""Upload a blood-smear cell image and see the model's prediction with a Grad-CAM heatmap.

    streamlit run app/streamlit_app.py

Educational demo only. Not a medical device and not for diagnosis.
"""
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.gradcam import make_gradcam_heatmap, overlay  # noqa: E402

st.set_page_config(page_title="Malaria cell classifier", layout="centered")
st.title("Malaria cell classifier")
st.caption(
    "Educational demo of a convolutional neural network trained on the NIH malaria "
    "cell images. **Not a diagnostic tool.**"
)

model_path = st.sidebar.text_input("Model file", "outputs/mobilenet.keras")
threshold = st.sidebar.slider(
    "Decision threshold", 0.05, 0.95, 0.5, 0.05,
    help="Lower it to catch more infected cells, at the cost of more false alarms.",
)


@st.cache_resource
def load(path):
    from tensorflow import keras
    return keras.models.load_model(path)


if not Path(model_path).exists():
    st.warning(f"No model at `{model_path}`. Train one first: `python -m src.train --model mobilenet`")
    st.stop()

model = load(model_path)
img_size = model.input_shape[1]

uploaded = st.file_uploader("Cell image (PNG or JPG)", type=["png", "jpg", "jpeg"])
if uploaded:
    image = np.asarray(Image.open(uploaded).convert("RGB").resize((img_size, img_size)), dtype=np.float32)
    heatmap, p = make_gradcam_heatmap(model, image)
    label = "Parasitized" if p >= threshold else "Uninfected"

    st.metric("Prediction", label, f"P(parasitized) = {p:.2f}", delta_color="off")
    left, right = st.columns(2)
    left.image(image.astype(np.uint8), caption="Input", width="stretch")
    right.image(overlay(image, heatmap), caption="Grad-CAM: where the model looked", width="stretch")
