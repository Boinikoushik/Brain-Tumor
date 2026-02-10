import numpy as np
import torch
import cv2

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# -----------------------------------------
# Automatically choose best layer per model
# -----------------------------------------
def get_target_layer(model, model_name):

    if model_name == "resnet50":
        return [model.layer4[-1]]

    elif model_name == "efficientnetv2":
        return [model.features[-1]]

    elif model_name in ["swin", "coatnet"]:
        # timm models — last feature block
        return [list(model.children())[-2]]

    else:
        return [list(model.children())[-1]]


# -----------------------------------------
# Generate GradCAM Heatmap
# -----------------------------------------
def generate_gradcam(model, model_name, input_tensor, image_rgb, pred_class):

    target_layers = get_target_layer(model, model_name)

    cam = GradCAM(model=model, target_layers=target_layers)

    targets = [ClassifierOutputTarget(pred_class)]

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets
    )[0]

    rgb_img = np.float32(image_rgb) / 255.0

    cam_image = show_cam_on_image(
        rgb_img,
        grayscale_cam,
        use_rgb=True
    )

    return cam_image
