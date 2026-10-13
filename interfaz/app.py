"""
Interfaz de usuario de DogVision (Streamlit)
Aplicación web para identificar la raza de un perro desde una imagen.
"""

import streamlit as st
from PIL import Image

from modelo.red_neuronal import CATEGORIAS, cargar_modelo_entrenado
from procesamiento.imagen import preprocesar_imagen_pil

RUTA_MODELO = "modelo/modelo_entrenado.pth"


def cargar_modelo():
    try:
        return cargar_modelo_entrenado(RUTA_MODELO)
    except FileNotFoundError:
        st.error(
            f"No se encontró el modelo entrenado en '{RUTA_MODELO}'. "
            "Ejecuta primero: python -m modelo.entrenamiento"
        )
        st.stop()


def main():
    st.set_page_config(
        page_title="DogVision",
        page_icon="🐕",
        layout="centered",
    )

    st.title("DogVision")
    st.subheader("Reconocimiento de Razas Caninas")
    st.write("Sube una imagen de un perro y el modelo te dirá su raza.")

    modelo = cargar_modelo()

    archivo = st.file_uploader(
        "Selecciona una imagen...", type=["jpg", "jpeg", "png", "bmp"]
    )

    if archivo is not None:
        imagen = Image.open(archivo)
        st.image(imagen, caption="Imagen cargada", use_container_width=True)

        with st.spinner("Analizando imagen..."):
            tensor = preprocesar_imagen_pil(imagen)

            import torch

            with torch.no_grad():
                salida = modelo(tensor)
                probabilidades = torch.softmax(salida, dim=1)
                indice_predicho = torch.argmax(probabilidades, dim=1).item()
                confianza = probabilidades[0][indice_predicho].item() * 100

        raza = CATEGORIAS[indice_predicho]

        st.success(f"**Predicción: {raza.replace('_', ' ').title()}** ({confianza:.1f}% de confianza)")

        with st.expander("Ver todas las probabilidades"):
            for i, cat in enumerate(CATEGORIAS):
                prob = probabilidades[0][i].item() * 100
                if prob > 0.1:
                    st.progress(prob / 100)
                    st.write(f"{cat.replace('_', ' ').title()}: {prob:.1f}%")


if __name__ == "__main__":
    main()