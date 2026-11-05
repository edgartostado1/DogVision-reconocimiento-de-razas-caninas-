"""
DogVision - Interfaz profesional minimalista
"""
import streamlit as st
from PIL import Image
import torch
from pathlib import Path

from modelo.red_neuronal import CATEGORIAS, cargar_modelo_entrenado
from procesamiento.imagen import preprocesar_imagen_pil
from database import obtener_raza

MODEL_PATH = "modelo/modelo_entrenado.pth"


def load_css():
    css_path = Path(__file__).parent / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_model():
    try:
        return cargar_modelo_entrenado(MODEL_PATH)
    except FileNotFoundError:
        return None


def format_breed(name: str) -> str:
    return name.replace("_", " ").title()


def render_ficha(raza_clave: str):
    """Renderiza la ficha de la raza predicha (datos desde SQLite)."""
    ficha = obtener_raza(raza_clave)

    if ficha is None:
        st.markdown('<div class="card ficha">No hay información de ficha para esta raza.</div>', unsafe_allow_html=True)
        return

    st.markdown(f"""
<div class="card ficha">
    <div class="ficha-title">Ficha de la raza</div>
    <h3 class="ficha-nombre">{ficha['nombre']}</h3>
    <div class="ficha-grid">
        <div class="ficha-item">
            <div class="ficha-label">Origen</div>
            <div class="ficha-value">{ficha['origen']}</div>
        </div>
        <div class="ficha-item">
            <div class="ficha-label">Tamaño</div>
            <div class="ficha-value">{ficha['tamano']}</div>
        </div>
        <div class="ficha-item">
            <div class="ficha-label">Peso</div>
            <div class="ficha-value">{ficha['peso']}</div>
        </div>
        <div class="ficha-item">
            <div class="ficha-label">Esperanza de vida</div>
            <div class="ficha-value">{ficha['esperanza_vida']}</div>
        </div>
    </div>
    <div class="ficha-bloque">
        <div class="ficha-label">Temperamento</div>
        <div class="ficha-texto">{ficha['temperamento']}</div>
    </div>
    <div class="ficha-bloque">
        <div class="ficha-label">Descripción</div>
        <div class="ficha-texto">{ficha['descripcion']}</div>
    </div>
</div>
""", unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="DogVision",
        page_icon="🐕",
        layout="centered",
    )
    load_css()

    logo_path = Path(__file__).parent / "assets" / "logo.png"
    col_izq, col_center, col_der = st.columns([1, 2, 1])
    with col_center:
        st.image(str(logo_path), use_container_width=True)
    st.markdown("""
<div class="hero-tagline">Identificación de razas caninas </div>
""", unsafe_allow_html=True)

    model = load_model()

    if model is None:
        st.markdown(f"""
<div class="card" style="text-align:center;">
    <h3 style="margin-top:0;">Modelo no encontrado</h3>
    <p style="color:var(--text-muted);">No existe <code>{MODEL_PATH}</code></p>
    <p style="color:var(--text-muted); font-size:.9rem;">Ejecuta:</p>
    <code style="background:var(--bg); padding:.5rem 1rem; border-radius:6px; display:inline-block;">python -m modelo.entrenamiento</code>
</div>
""", unsafe_allow_html=True)
        return

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Subir imagen")
    st.markdown("""
<div class="upload-zone">
    <div class="upload-icon">📁</div>
    <h3>Arrastra una imagen o haz clic</h3>
    <p>JPG, PNG, BMP · Máx. 200MB</p>
</div>
""", unsafe_allow_html=True)
    uploaded = st.file_uploader(" ", type=["jpg", "jpeg", "png", "bmp"], label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    if not uploaded:
        st.markdown('<div class="card" style="text-align:center; color:var(--text-muted);">Sube una foto de un perro para empezar</div>', unsafe_allow_html=True)
        return

    img = Image.open(uploaded)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Vista previa")
    st.markdown('<div class="img-wrap">', unsafe_allow_html=True)
    st.image(img, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="img-info">{img.width}×{img.height} · {img.mode} · {img.format or "—"}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    with st.spinner("Analizando..."):
        tensor = preprocesar_imagen_pil(img)
        with torch.no_grad():
            out = model(tensor)
            probs = torch.softmax(out, dim=1)
            idx = torch.argmax(probs, dim=1).item()
            conf = probs[0][idx].item() * 100

    breed = CATEGORIAS[idx]

    st.markdown(f"""
<div class="card result">
    <div class="result-badge">Raza identificada</div>
    <h2 class="breed-name">{format_breed(breed)}</h2>
    <div class="meter"><div class="meter-fill" style="width:{conf:.1f}%"></div></div>
    <div class="meter-labels">
        <span class="label">Confianza del modelo</span>
        <span class="value">{conf:.1f}%</span>
    </div>
</div>
""", unsafe_allow_html=True)

    # Ficha de la raza (SQLite)
    render_ficha(breed)

    top5 = torch.topk(probs, 5)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Top 5 predicciones")
    st.markdown('<div class="top-list">', unsafe_allow_html=True)
    for rank, (i, p) in enumerate(zip(top5.indices[0].tolist(), top5.values[0].tolist()), 1):
        cls = "rank-1" if rank == 1 else "rank-2" if rank == 2 else "rank-3" if rank == 3 else "other"
        st.markdown(f"""
<div class="top-item">
    <span class="top-rank {cls}">{rank}</span>
    <div class="top-info">
        <div class="top-breed">{format_breed(CATEGORIAS[i])}</div>
        <div class="top-bar"><div class="top-fill" style="width:{p*100:.1f}%"></div></div>
    </div>
    <span class="top-pct">{p*100:.1f}%</span>
</div>
""", unsafe_allow_html=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

    with st.expander("Ver las 120 probabilidades"):
        all_p = probs[0].tolist()
        sorted_idx = sorted(range(len(all_p)), key=lambda i: all_p[i], reverse=True)
        cols = st.columns(3)
        for ci, i in enumerate(sorted_idx):
            p = all_p[i] * 100
            if p < 0.1: continue
            with cols[ci % 3]:
                st.markdown(f"""
<div style="padding:.5rem; background:var(--bg); border-radius:8px; border:1px solid var(--border); margin-bottom:.5rem;">
    <div style="font-weight:600; font-size:.85rem;">{format_breed(CATEGORIAS[i])}</div>
    <div style="display:flex; gap:.5rem; margin-top:.25rem;">
        <div style="flex:1; height:4px; background:var(--border); border-radius:2px; overflow:hidden;">
            <div style="width:{p:.1f}%; height:100%; background:linear-gradient(90deg,var(--primary),#3b82f6);"></div>
        </div>
        <span style="font-size:.75rem; font-weight:600; color:var(--primary); min-width:40px;">{p:.2f}%</span>
    </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="footer">

</div>
""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()