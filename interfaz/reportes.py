"""
Generacion de reportes PDF de predicciones.
Usa ReportLab (https://www.reportlab.com/).
"""
from io import BytesIO
from datetime import datetime
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage,
)

LOGO_PATH = Path(__file__).parent / "assets" / "logo.png"


def generar_reporte(
    imagen_pil,
    raza_clave: str,
    nombre_raza: str,
    confianza: float,
    probs_top5: list,
    ficha: dict | None,
) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("title", parent=styles["Title"], fontSize=20, spaceAfter=6)
    h_style = ParagraphStyle("h", parent=styles["Heading3"], fontSize=13, spaceBefore=10, spaceAfter=4)
    normal = styles["Normal"]

    story = []

    # Encabezado con logo (izquierda)
    header_cells = []
    if LOGO_PATH.exists():
        logo_img = RLImage(str(LOGO_PATH), width=2.5 * cm, height=2.5 * cm)
        header_cells.append(logo_img)
    header_cells.append(Paragraph("DogVision - Reporte de Prediccion", title_style))
    header_table = Table([header_cells], colWidths=[2.5 * cm, 13.5 * cm])
    header_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(header_table)
    story.append(Paragraph(
        f"Generado el {datetime.now().strftime('%d/%m/%Y %H:%M')} - Modelo ResNet18 (120 razas)",
        ParagraphStyle("sub", parent=normal, fontSize=9, textColor=colors.grey),
    ))
    story.append(Spacer(1, 0.4 * cm))

    # Imagen analizada
    story.append(Paragraph("Imagen analizada", h_style))
    img_buf = BytesIO()
    imagen_pil.save(img_buf, format="PNG")
    img_buf.seek(0)
    rl_img = RLImage(img_buf, width=5 * cm, height=5 * cm)
    story.append(rl_img)
    story.append(Spacer(1, 0.3 * cm))

    # Resultado
    story.append(Paragraph("Resultado", h_style))
    res_data = [
        ["Raza predicha", nombre_raza],
        ["Confianza", f"{confianza:.1f}%"],
    ]
    res_table = Table(res_data, colWidths=[4 * cm, 12 * cm])
    res_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e8f0fe")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 11),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(res_table)
    story.append(Spacer(1, 0.3 * cm))

    # Top 5
    story.append(Paragraph("Top 5 predicciones", h_style))
    top_data = [["#", "Raza", "Probabilidad"]]
    for rank, item in enumerate(probs_top5, 1):
        top_data.append([str(rank), item["breed"], f"{item['prob']:.1f}%"])
    top_table = Table(top_data, colWidths=[1.5 * cm, 10.5 * cm, 4 * cm])
    top_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c7be5")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (0, 0), (0, -1), "CENTER"),
        ("ALIGN", (2, 0), (2, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(top_table)
    story.append(Spacer(1, 0.3 * cm))

    # Ficha de la raza
    if ficha:
        story.append(Paragraph("Ficha de la raza", h_style))
        cell_style = ParagraphStyle("cell", parent=normal, fontSize=10, leading=12)
        ficha_data = [
            ["Origen", ficha.get("origen", "-")],
            ["Tamano", ficha.get("tamano", "-")],
            ["Peso", ficha.get("peso", "-")],
            ["Esperanza de vida", ficha.get("esperanza_vida", "-")],
            ["Temperamento", Paragraph(ficha.get("temperamento", "-"), cell_style)],
            ["Descripcion", Paragraph(ficha.get("descripcion", "-"), cell_style)],
        ]
        ficha_table = Table(ficha_data, colWidths=[4 * cm, 12 * cm])
        ficha_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f0f7ff")),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(ficha_table)
        story.append(Spacer(1, 0.3 * cm))

    # Pie
    story.append(Paragraph(
        "DogVision v2.0 - ResNet18 - PyTorch + Streamlit - Generado automaticamente",
        ParagraphStyle("footer", parent=normal, fontSize=8, textColor=colors.grey, alignment=1),
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()