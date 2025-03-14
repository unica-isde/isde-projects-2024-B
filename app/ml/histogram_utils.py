"""
This generates the three color histograms of an image. It accepts an url of an
image and returns the three histograms.
"""
import importlib
import json
import logging
import os
import torch
from PIL import Image
from torchvision import transforms
import cv2 as cv

from app.config import Configuration


conf = Configuration()


def fetch_image(image_id):
    """Gets the image from the specified ID. It returns only images
    downloaded in the folder specified in the configuration object."""
    image_path = os.path.join(conf.image_folder_path, image_id)
    img = Image.open(image_path)
    return img


def histogram(img_id):
    """Return the histograms for the three color (R, G, B) channel of the image."""
    img_path = f'app/static/imagenet_subset/{img_id}'
    img = cv.imread(img_path)

    if img is None:
        raise ValueError(f"Could not load image at path: {img_path}")

    if img.size == 0:
        raise ValueError(f"The image at path: {img_path} is empty or invalid")

    histr_r = cv.calcHist([img], [2], None, [256], [0, 256])
    histr_r = histr_r.flatten().tolist()
    histr_g = cv.calcHist([img], [1], None, [256], [0, 256])
    histr_g = histr_g.flatten().tolist()
    histr_b = cv.calcHist([img], [0], None, [256], [0, 256])
    histr_b = histr_b.flatten().tolist()

    return histr_b, histr_g, histr_r
