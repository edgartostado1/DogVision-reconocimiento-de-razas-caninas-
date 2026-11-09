"""
DogVision - Interfaz profesional minimalista
"""
import streamlit as st
from PIL import Image
import torch
from pathlib import Path

from modelo.red_neuronal import CATEGORIAS, cargar_modelo_entrenado
from procesamiento.imagen import preprocesar_imagen_pil
from database import obtener_raza, obtener_todas
from interfaz.reportes import generar_reporte

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


@st.cache_data(show_spinner=False)
def get_famous_images():
    famous = [
        "golden_retriever", "siberian_husky", "french_bulldog", "german_shepherd",
        "border_collie", "labrador_retriever", "pomeranian", "chihuahua",
        "beagle", "pug", "rottweiler", "boxer", "samoyed", "doberman",
        "mexican_hairless",
    ]
    items = []
    train_dir = Path("data/train")
    if not train_dir.exists():
        return items
    for breed in famous:
        breed_dir = train_dir / breed
        if not breed_dir.is_dir():
            continue
        imgs = list(breed_dir.glob("*.jpg"))
        if imgs:
            items.append((breed, str(imgs[0])))
    return items


def render_ficha(raza_clave: str):
    ficha = obtener_raza(raza_clave)

    if ficha is None:
        st.markdown('<div class="card ficha">No hay informacion de ficha para esta raza.</div>', unsafe_allow_html=True)
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
            <div class="ficha-label">Tamano</div>
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
        <div class="ficha-label">Descripcion</div>
        <div class="ficha-texto">{ficha['descripcion']}</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# NAVEGACION POR PAGINAS (sidebar)
# ============================================================
PAGINAS = ["Inicio", "Reporte PDF", "Catalogo de razas", "Nosotros"]


def sidebar_nav():
    with st.sidebar:
        st.markdown("## DogVision")
        st.markdown("---")
        pagina = st.radio("Navegacion", PAGINAS, label_visibility="collapsed")
        st.markdown("---")
        st.caption("DogVision - ResNet18 - PyTorch")
    return pagina


# ============================================================
# PAGINA 1: INICIO (prediccion)
# ============================================================
def pagina_inicio(model):
    st.markdown("""
<div class="hero-tagline">Identificacion de razas caninas con Deep Learning</div>
""", unsafe_allow_html=True)

    if model is None:
        st.markdown(f"""
<div class="card" class="text-center">
    <h3 class="mt-0">Modelo no encontrado</h3>
    <p class="text-muted">No existe <code>{MODEL_PATH}</code></p>
    <p class="text-muted-sm">Ejecuta:</p>
    <code class="code-block">python -m modelo.entrenamiento</code>
</div>
""", unsafe_allow_html=True)
        return

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Subir imagen")
    st.markdown("""
<div class="upload-zone" id="upload">
    <div class="upload-icon">IMG</div>
    <h3>Arrastra una imagen o haz clic</h3>
    <p>JPG, PNG, BMP - Max. 200MB</p>
</div>
""", unsafe_allow_html=True)
    uploaded = st.file_uploader(" ", type=["jpg", "jpeg", "png", "bmp"], label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    if not uploaded:
        famous_items = get_famous_images()
        if famous_items:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Razas populares")
            if "famous_idx" not in st.session_state:
                st.session_state["famous_idx"] = 0
            fi = st.session_state["famous_idx"] % len(famous_items)
            fb, fp = famous_items[fi]
            col_img, col_info = st.columns([2, 1])
            with col_img:
                st.image(fp, width=260)
            with col_info:
                st.markdown(f'<div class="famous-name">{format_breed(fb)}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="img-info">{fi + 1} / {len(famous_items)}</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Anterior", key="famous_prev"):
                    st.session_state["famous_idx"] = (fi - 1) % len(famous_items)
                    st.rerun()
            with c2:
                if st.button("Siguiente", key="famous_next"):
                    st.session_state["famous_idx"] = (fi + 1) % len(famous_items)
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card" class="text-center-muted">Sube una foto de un perro para empezar</div>', unsafe_allow_html=True)
        return

    img = Image.open(uploaded)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Vista previa")
    st.markdown('<div class="img-wrap">', unsafe_allow_html=True)
    st.image(img, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="img-info">{img.width}x{img.height} - {img.mode} - {img.format or "-"} </div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    with st.spinner("Analizando..."):
        tensor = preprocesar_imagen_pil(img)
        with torch.no_grad():
            out = model(tensor)
            probs = torch.softmax(out, dim=1)
            idx = torch.argmax(probs, dim=1).item()
            conf = probs[0][idx].item() * 100

    breed = CATEGORIAS[idx]
    nombre_raza = format_breed(breed)

    st.markdown(f"""
<div class="card result">
    <div class="result-badge">Raza identificada</div>
    <h2 class="breed-name">{nombre_raza}</h2>
    <div class="meter"><div class="meter-fill" style="width:{conf:.1f}%"></div></div>
    <div class="meter-labels">
        <span class="label">Confianza del modelo</span>
        <span class="value">{conf:.1f}%</span>
    </div>
</div>
""", unsafe_allow_html=True)

    render_ficha(breed)

    top5 = torch.topk(probs, 5)

    st.session_state["ultima_pred"] = {
        "img": img, "breed": breed, "nombre": nombre_raza,
        "conf": conf, "probs": probs, "top5": top5,
    }

    top5_list = [
        {"breed": format_breed(CATEGORIAS[i]), "prob": p * 100}
        for i, p in zip(top5.indices[0].tolist(), top5.values[0].tolist())
    ]
    ficha = obtener_raza(breed)
    pdf_bytes = generar_reporte(img, breed, nombre_raza, conf, top5_list, ficha)

    st.download_button(
        label="Descargar reporte (PDF)",
        data=pdf_bytes,
        file_name=f"dogvision_{breed}_{int(conf)}.pdf",
        mime="application/pdf",
    )

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
<div class="prob-item">
    <div class="prob-name">{format_breed(CATEGORIAS[i])}</div>
    <div class="prob-row">
        <div class="prob-bar">
            <div style="width:{p:.1f}%; height:100%; background:linear-gradient(90deg,var(--primary),#3b82f6);"></div>
        </div>
        <span class="prob-pct">{p:.2f}%</span>
    </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="footer">
    DogVision - ResNet18 - PyTorch + Streamlit
</div>
""", unsafe_allow_html=True)


# ============================================================
# PAGINA 2: REPORTE PDF
# ============================================================
def pagina_reporte():
    st.markdown("""
<div class="hero-tagline">Reporte PDF de la ultima prediccion</div>
""", unsafe_allow_html=True)

    pred = st.session_state.get("ultima_pred")
    if not pred:
        st.markdown('<div class="card" class="text-center-muted">Aun no has hecho ninguna prediccion. Ve a "Inicio" y sube una imagen.</div>', unsafe_allow_html=True)
        return

    img = pred["img"]
    breed = pred["breed"]
    conf = pred["conf"]
    top5 = pred["top5"]
    nombre_raza = pred["nombre"]

    st.markdown(f'<div class="card"><h3 class="mt-0">Ultima prediccion: {nombre_raza} ({conf:.1f}%)</h3></div>', unsafe_allow_html=True)

    top5_list = [
        {"breed": format_breed(CATEGORIAS[i]), "prob": p * 100}
        for i, p in zip(top5.indices[0].tolist(), top5.values[0].tolist())
    ]
    ficha = obtener_raza(breed)
    pdf_bytes = generar_reporte(img, breed, nombre_raza, conf, top5_list, ficha)

    st.download_button(
        label="Descargar reporte (PDF)",
        data=pdf_bytes,
        file_name=f"dogvision_{breed}_{int(conf)}.pdf",
        mime="application/pdf",
    )

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Vista previa del contenido")
    st.image(img, width=200)
    st.markdown(f"**Raza:** {nombre_raza}")
    st.markdown(f"**Confianza:** {conf:.1f}%")
    st.markdown("**Top 5:**")
    for rank, (i, p) in enumerate(zip(top5.indices[0].tolist(), top5.values[0].tolist()), 1):
        st.markdown(f"{rank}. {format_breed(CATEGORIAS[i])} - {p*100:.1f}%")
    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# PAGINA 3: CATALOGO DE RAZAS
# ============================================================
def pagina_catalogo():
    st.markdown("""
<div class="hero-tagline">Catalogo de las 120 razas del modelo</div>
""", unsafe_allow_html=True)

    razas = obtener_todas()
    if not razas:
        st.markdown('<div class="card" class="text-center-muted">No hay datos. Ejecuta: python -m database.crear_db</div>', unsafe_allow_html=True)
        return

    nombres = [f"{r['nombre']} ({r['clave']})" for r in razas]
    seleccion = st.selectbox("Buscar raza", ["- selecciona -"] + nombres)

    if seleccion != "- selecciona -":
        clave = seleccion.split("(")[-1].rstrip(")")
        ficha = obtener_raza(clave)
        if ficha:
            st.markdown(f'<div class="card"><h2 class="breed-name">{ficha["nombre"]}</h2></div>', unsafe_allow_html=True)
            render_ficha(clave)

    st.markdown("### Todas las razas")
    cols = st.columns(3)
    for ci, r in enumerate(razas):
        with cols[ci % 3]:
            st.markdown(f"""
<div class="breed-chip">
    <b>{r['nombre']}</b><br><span class="text-muted-xs">{r['origen']}</span>
</div>
""", unsafe_allow_html=True)


# ============================================================
# PAGINA 4: NOSOTROS
# ============================================================
def pagina_nosotros():
    st.markdown('<div class="hero-tagline">Nosotros</div>', unsafe_allow_html=True)

    html = (
        '<div class="card">'
        '<h3 class="mt-0">Equipo DogVision</h3>'
        '<p>Somos un equipo apasionado por la inteligencia artificial aplicada a la vision por computadora. '
        'Desarrollamos DogVision como proyecto educativo para demostrar como el Deep Learning puede '
        'identificar razas caninas a partir de una sola imagen.</p>'
        '<h4>Mision</h4>'
        '<p>Acercar el reconocimiento de imagenes con IA a cualquier persona, de forma gratuita y accesible, '
        'usando tecnologias open source y un dataset publico.</p>'
        '<h4>Valores</h4>'
        '<ul>'
        '<li><b>Open Source:</b> codigo y datos abiertos para la comunidad.</li>'
        '<li><b>Educativo:</b> cada modulo esta documentado para aprender.</li>'
        '<li><b>Practico:</b> una app que funciona de verdad, no solo un demo.</li>'
        '</ul>'
        '<h4>Contacto</h4>'
        '<p>GitHub: <a href="https://github.com" target="_blank">DogVision-reconocimiento-de-razas-caninas</a></p>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# ============================================================
# MAIN
# ============================================================
def main():
    st.set_page_config(
        page_title="DogVision",
        page_icon="D",
        layout="centered",
    )
    load_css()

    logo_path = Path(__file__).parent / "assets" / "logo.png"
    if logo_path.exists():
        col_l, col_c, col_r = st.columns([1, 2, 1])
        with col_c:
            st.image(str(logo_path), use_container_width=True)

    pagina = sidebar_nav()

    if pagina == "Inicio":
        pagina_inicio(load_model())
    elif pagina == "Reporte PDF":
        pagina_reporte()
    elif pagina == "Catalogo de razas":
        pagina_catalogo()
    elif pagina == "Nosotros":
        pagina_nosotros()


if __name__ == "__main__":
    main()