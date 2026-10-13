"""
Preprocesamiento de imágenes para el modelo de Deep Learning
"""

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

TAMANO_IMAGEN = 224

NORMALIZACION = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


def cargar_imagen(ruta: str) -> np.ndarray:
    imagen = cv2.imread(ruta)
    if imagen is None:
        raise ValueError(f"No se pudo cargar la imagen: {ruta}")
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    return imagen


def preprocesar_imagen(imagen: np.ndarray) -> torch.Tensor:
    transformacion = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((TAMANO_IMAGEN, TAMANO_IMAGEN)),
        transforms.ToTensor(),
        NORMALIZACION,
    ])
    tensor = transformacion(imagen)
    return tensor.unsqueeze(0)


def preprocesar_imagen_pil(imagen_pil: Image.Image) -> torch.Tensor:
    transformacion = transforms.Compose([
        transforms.Resize((TAMANO_IMAGEN, TAMANO_IMAGEN)),
        transforms.ToTensor(),
        NORMALIZACION,
    ])
    tensor = transformacion(imagen_pil)
    return tensor.unsqueeze(0)