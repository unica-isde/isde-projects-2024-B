"""
This generates the enhanced image from an image ID.
"""
import importlib
import json
import logging
import os

import PIL.ImageEnhance
import torch
from PIL import Image
from torchvision import transforms
from PIL import ImageEnhance

from app.config import Configuration


conf = Configuration()


def fetch_image(image_id):
    """Gets the image from the specified ID. It returns only images
    downloaded in the folder specified in the configuration object."""
    image_path = os.path.join(conf.image_folder_path, image_id)
    img = Image.open(image_path)
    return img

def enhance_image(image, color_factor,brightness_factor, contrast_factor, sharpness_factor):
    '''This function enhances the image with the specified factors'''
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(color_factor)

    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness_factor)

    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast_factor)

    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(sharpness_factor)

    return image

def transform_image(img_id,color,brightness,contrast,sharpness):
    '''This function enhances and saves the image with the specified factors'''
    enhanced_image_path = f"app/static/enhanced_images"
    os.makedirs(enhanced_image_path, exist_ok=True)
    img = fetch_image(img_id)
    try:
        new_img = enhance_image(img, color, brightness, contrast, sharpness)
        new_img.save(os.path.join(enhanced_image_path, f"enhanced_{img_id}"))
    except Exception as e:
        print(f"Error during image enhancement: {e}")
        raise e



