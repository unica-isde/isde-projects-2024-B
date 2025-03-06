"""
This is a simple classification service. It accepts an url of an
image and returns the top-5 classification labels and scores.
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
    """Classifica un'immagine caricata come dati binari."""
    # Carica l'immagine dai dati binari
    img = Image.open(BytesIO(img_data))

    # Carica il modello specificato
    model = get_model(model_id)  # Usa la funzione esistente per caricare il modello
    model.eval()  # Modalità di valutazione del modello

    # Trasformazioni dell'immagine
    transform = transforms.Compose(
        [
            transforms.Resize(256),  # Ridimensiona il lato più corto a 256 pixel
            transforms.CenterCrop(224),  # Ritaglia al centro per ottenere 224x224
            transforms.ToTensor(),  # Converte l'immagine in un tensore
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # Normalizza i valori dei pixel
        ]
    )

    # Prepara l'immagine per il modello
    img = img.convert("RGB")  # Assicura che l'immagine sia in formato RGB
    preprocessed = transform(img).unsqueeze(0)  # Aggiunge una dimensione per il batch

    # Ottieni l'output del modello
    out = model(preprocessed)
    _, indices = torch.sort(out, descending=True)

    # Converte le probabilità in percentuali
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

    # Ottieni le etichette
    labels = get_labels()

    # Prendi le prime 5 predizioni
    output = [[labels[idx], percentage[idx].item()] for idx in indices[0][:5]]

    # Chiudi l'immagine per liberare memoria
    img.close()

    return output
