import streamlit as st
import torch
import numpy as np
import cv2
import os
import gdown
from PIL import Image

from models import get_model
from transforms import get_transform

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# ======================
# Page Config
# ======================
st.set_page_config(page_title="Brain Tumor AI", layout="centered")
st.title("🧠 Brain Tumor Classification")

device = torch.device("cpu")


# ======================
# Available Models (Drive Links)
# ======================
MODEL_FILES = {
    "ResNet50": {
        "path": "best_resnet50.pth",
        "url": "https://drive.google.com/file/d/1ljTeAMxeon4ao2uYMe_Tx_PshnhV8mL9"
    },
    "Swin Transformer": {
        "path": "best_swin.pth",
        "url": "https://drive.google.com/file/d/1rMmFKLILvzm1G36wUhNo-HFKiTpF6ML-"
    },
    "CoAtNet": {
        "path": "best_coatnet.pth",
        "url": "https://drive.google.com/file/d/1jm2isXCBT8isRKH1r6s8BLUnx-y8Ev93"
    },
    "EfficientNetV2": {
        "path": "efficientnetv2_streamlit.pth",
        "url": "https://drive.google.com/file/d/1TyMlu6yCpNpfE_xcZLfhcZqRLDMbWNMV"
    }
}


# ======================
# Download Model If Needed
# ======================
@st.cache_resource
def ensure_model(model_info):
    path = model_info["path"]
    url = model_info["url"]

    # 🚨 Delete bad file if it exists
    if os.path.exists(path):
        os.remove(path)

    with st.spinner("⬇️ Downloading model from Google Drive..."):
        gdown.download(url, path, fuzzy=True, quiet=False)


# ======================
# Select GradCAM Layer Automatically
# ======================
def get_target_layer(model, model_name):

    if model_name == "resnet50":
        return [model.layer4[-1]]

    elif model_name == "efficientnetv2":
        return [model.features[-1]]

    elif model_name in ["swin", "coatnet"]:
        return [list(model.children())[-2]]

    else:
        return [list(model.children())[-1]]


# ======================
# Model Loader (cached)
# ======================
@st.cache_resource
def load_model(pth_path):

    checkpoint = torch.load(pth_path, map_location="cpu")

    # NEW STREAMLIT FORMAT
    if isinstance(checkpoint, dict) and "model_name" in checkpoint:
        model_name = checkpoint["model_name"]
        num_classes = checkpoint["num_classes"]
        classes = checkpoint["classes"]
        state_dict = checkpoint["model_state"]

    # OLD FORMAT
    else:
        st.warning("⚠️ Old .pth format detected")

        model_name = "resnet50"
        num_classes = 4
        classes = ["glioma","meningioma","notumor","pituitary"]
        state_dict = checkpoint

    model = get_model(model_name, num_classes)
    model.load_state_dict(state_dict)
    model.eval()

    return model, classes, model_name


# ======================
# UI Controls
# ======================
selected_model = st.selectbox(
    "Select Model",
    list(MODEL_FILES.keys())
)

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg","png","jpeg"]
)

show_cam = st.checkbox("🔥 Show Grad-CAM Explanation")

predict_button = st.button("🔍 Predict")


# ======================
# Prediction
# ======================
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI", use_container_width=True)

    if predict_button:

        with st.spinner("Running AI model..."):

            transform = get_transform()
            input_tensor = transform(image).unsqueeze(0).to(device)

            # 🔥 NEW PART: ensure model downloaded
            model_info = MODEL_FILES[selected_model]
            ensure_model(model_info)

            model, classes, model_name = load_model(model_info["path"])

            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.softmax(outputs, dim=1)
                pred = torch.argmax(probs, dim=1).item()
                confidence = probs[0][pred].item()

        st.success(f"Prediction: {classes[pred]}")
        st.write(f"Confidence: {confidence*100:.2f}%")

        # ======================
        # GRAD-CAM
        # ======================
        if show_cam:

            image_np = np.array(image)
            image_np = cv2.resize(image_np, (224,224))

            target_layers = get_target_layer(model, model_name)

            cam = GradCAM(model=model, target_layers=target_layers)

            targets = [ClassifierOutputTarget(pred)]

            grayscale_cam = cam(
                input_tensor=input_tensor,
                targets=targets
            )[0]

            rgb_img = np.float32(image_np) / 255.0

            cam_image = show_cam_on_image(
                rgb_img,
                grayscale_cam,
                use_rgb=True
            )

            st.image(cam_image, caption="Grad-CAM Explanation")
