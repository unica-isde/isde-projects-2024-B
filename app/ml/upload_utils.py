"""
This is the upload_utils module, which contains
utility functions for the upload route.
"""

import importlib
import json
import logging
import os
import torch
from PIL import Image
from torchvision import transforms
from io import BytesIO
from .classification_utils import get_model, get_labels

from app.config import Configuration

conf = Configuration()


def uploaded_image(model_id, img_data):
    """ This is a function take the uploaded image and classify it using the model"""
    # Load the image
    img = Image.open(BytesIO(img_data))

    # Load the model
    model = get_model(model_id)
    model.eval()

    # Image transformation
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    # Prepare the image for the model
    img = img.convert("RGB")  # Convert the image in RGB
    preprocessed = transform(img).unsqueeze(0)

    out = model(preprocessed)
    _, indices = torch.sort(out, descending=True)

    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

    labels = get_labels()

    output = [[labels[idx], percentage[idx].item()] for idx in indices[0][:5]]

    img.close()

    return output
