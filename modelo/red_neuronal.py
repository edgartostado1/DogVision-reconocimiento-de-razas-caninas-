"""
Red Neuronal Convolucional (CNN) para clasificación de razas caninas
Transfer Learning con ResNet18 preentrenado en ImageNet.
Dataset: Stanford Dogs (120 razas, ~20,000 imágenes)
"""

import torch
import torch.nn as nn
import torchvision.models as models

# 120 razas de Stanford Dogs en el ORDEN EXACTO de las carpetas en data/train/ y data/val/
CATEGORIAS = [
    "affenpinscher",
    "afghan_hound",
    "african_hunting_dog",
    "airedale",
    "american_staffordshire_terrier",
    "appenzeller",
    "australian_terrier",
    "basenji",
    "basset",
    "beagle",
    "bedlington_terrier",
    "bernese_mountain_dog",
    "black_and_tan_coonhound",
    "blenheim_spaniel",
    "bloodhound",
    "bluetick",
    "border_collie",
    "border_terrier",
    "borzoi",
    "boston_bull",
    "bouvier_des_flandres",
    "boxer",
    "brabancon_griffon",
    "briard",
    "brittany_spaniel",
    "bull_mastiff",
    "cairn",
    "cardigan",
    "chesapeake_bay_retriever",
    "chihuahua",
    "chow",
    "clumber",
    "cocker_spaniel",
    "collie",
    "curly_coated_retriever",
    "dandie_dinmont",
    "dhole",
    "dingo",
    "doberman",
    "english_foxhound",
    "english_setter",
    "english_springer",
    "entlebucher",
    "eskimo_dog",
    "flat_coated_retriever",
    "french_bulldog",
    "german_shepherd",
    "german_short_haired_pointer",
    "giant_schnauzer",
    "golden_retriever",
    "gordon_setter",
    "great_dane",
    "great_pyrenees",
    "greater_swiss_mountain_dog",
    "groenendael",
    "ibizan_hound",
    "irish_setter",
    "irish_terrier",
    "irish_water_spaniel",
    "irish_wolfhound",
    "italian_greyhound",
    "japanese_spaniel",
    "keeshond",
    "kelpie",
    "kerry_blue_terrier",
    "komondor",
    "kuvasz",
    "labrador_retriever",
    "lakeland_terrier",
    "leonberg",
    "lhasa",
    "malamute",
    "malinois",
    "maltese_dog",
    "mexican_hairless",
    "miniature_pinscher",
    "miniature_poodle",
    "miniature_schnauzer",
    "newfoundland",
    "norfolk_terrier",
    "norwegian_elkhound",
    "norwich_terrier",
    "old_english_sheepdog",
    "otterhound",
    "papillon",
    "pekinese",
    "pembroke",
    "pomeranian",
    "pug",
    "redbone",
    "rhodesian_ridgeback",
    "rottweiler",
    "saint_bernard",
    "saluki",
    "samoyed",
    "schipperke",
    "scotch_terrier",
    "scottish_deerhound",
    "sealyham_terrier",
    "shetland_sheepdog",
    "shih_tzu",
    "siberian_husky",
    "silky_terrier",
    "soft_coated_wheaten_terrier",
    "staffordshire_bullterrier",
    "standard_poodle",
    "standard_schnauzer",
    "sussex_spaniel",
    "tibetan_mastiff",
    "tibetan_terrier",
    "toy_poodle",
    "toy_terrier",
    "vizsla",
    "walker_hound",
    "weimaraner",
    "welsh_springer_spaniel",
    "west_highland_white_terrier",
    "whippet",
    "wire_haired_fox_terrier",
    "yorkshire_terrier",
]

TAMANO_IMAGEN = 224


def crear_modelo(num_clases: int = 120, freeze_backbone: bool = True) -> nn.Module:
    """Crea un ResNet18 con Transfer Learning para clasificación de razas caninas."""
    modelo = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in modelo.parameters():
            param.requires_grad = False

    modelo.fc = nn.Linear(modelo.fc.in_features, num_clases)
    return modelo


def cargar_modelo_entrenado(ruta_pesos: str, num_clases: int = 120) -> nn.Module:
    """Carga un modelo previamente entrenado."""
    modelo = crear_modelo(num_clases=num_clases, freeze_backbone=False)
    modelo.load_state_dict(torch.load(ruta_pesos, map_location="cpu"))
    modelo.eval()
    return modelo