# models.py

import timm
import torch.nn as nn
import torchvision.models as models

def get_model(name, num_classes):

    if name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif name == "efficientnetv2":
        model = models.efficientnet_v2_s(weights=None)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            num_classes
        )

    elif name == "swin":
        model = timm.create_model(
            "swin_base_patch4_window7_224",
            pretrained=False,
            num_classes=num_classes
        )

    elif name == "coatnet":
        model = timm.create_model(
            "coatnet_0_rw_224",
            pretrained=False,
            num_classes=num_classes
        )

    else:
        raise ValueError(f"Unknown model {name}")

    return model
