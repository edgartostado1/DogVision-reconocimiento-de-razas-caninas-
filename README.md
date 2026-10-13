# DogVision: Reconocimiento de Razas Caninas

Aplicación en Python que utiliza Deep Learning para reconocer la raza de un perro mediante una imagen.

## Dataset

**Stanford Dogs Dataset** - 120 razas de perros, ~20,000 imágenes
- Entrenamiento: ~16,400 imágenes
- Validación: ~4,100 imágenes

## Tecnologías

- Python
- PyTorch / Torchvision
- OpenCV
- Pillow
- NumPy
- Matplotlib
- Streamlit

## Estructura del proyecto

```
DogVision/
├── modelo/              # Definición del modelo y entrenamiento
├── procesamiento/       # Preprocesamiento de imágenes
├── interfaz/            # Interfaz de usuario (Streamlit)
├── app.py               # Punto de entrada
├── requirements.txt     # Dependencias
├── data/                # Dataset local (no se sube a GitHub)
│   ├── train/           # 120 razas, ~16,400 imágenes
│   └── val/             # 120 razas, ~4,100 imágenes
└── README.md
```

## Flujo de la aplicación

```
Imagen → Procesamiento → Modelo de Deep Learning (ResNet18) → Predicción de raza
```

## Instalación

```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

## Uso

```bash
# Entrenar el modelo (primera vez)
python -m modelo.entrenamiento

# Ejecutar la aplicación
streamlit run interfaz/app.py
# O desde la raíz:
python app.py
```

## Modelo

- **Arquitectura**: ResNet18 (Transfer Learning desde ImageNet)
- **Clases**: 120 razas de perros
- **Tamaño de entrada**: 224x224
- **Optimizador**: Adam (lr=0.0001)
- **Scheduler**: ReduceLROnPlateau

## Equipo

Proyecto desarrollado para reconocimiento de razas caninas.