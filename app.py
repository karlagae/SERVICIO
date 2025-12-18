import cv2
import numpy as np
from PIL import Image
import streamlit as st
import easyocr
import re
from spellchecker import SpellChecker

# =============================
#  OCR con EasyOCR (cacheado)
# =============================
@st.cache_resource
def get_ocr_reader():
    return easyocr.Reader(['es'], gpu=False)

reader = get_ocr_reader()

def ocr_easy_detail(img_bgr):
    """EasyOCR con cajas: lista de (bbox, text, conf)"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return reader.readtext(img_rgb, detail=1)

def _bbox_to_xywh(bbox):
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
    return x1, y1, x2 - x1, y2 - y1

def _norm_text(s):
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

# =============================
#  Autocorrector (lo dejo por si lo quieres)
# =============================
@st.cache_resource
def get_spellchecker():
    return SpellChecker(language="es")

spell = get_spellchecker()

def autocorregir_texto(texto: str):
    if not texto or not texto.strip():
        return texto, []

    patron = r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+"
    tokens = re.findall(patron, texto)
    desconocidas = spell.unknown([t.lower() for t in tokens])

    cambios = []

    def corregir_match(m):
        palabra = m.group(0)
        lower = palabra.lower()
        if lower in desconocidas:
            sugerida = spell.correction(lower) or lower
            if palabra.isupper():
                sugerida = sugerida.upper()
            elif palabra[0].isupper():
                sugerida = sugerida.capitalize()
            if sugerida.lower() != lower:
                cambios.append((palabra, sugerida))
            return sugerida
        return palabra

    texto_corregido = re.sub(patron, corregir_match, texto)
    return texto_corregido, cambios

# =============================
#  1) Detectar checkboxes (cuadritos) - ROBUSTO para CIRCULACIÓN
# =============================
def detectar_checkboxes(crop_bgr):
    """
    Detecta cuadritos tipo checkbox dentro de un recorte (CIRCULACIÓN).
    Devuelve lista de (x,y,w,h).
    """
    H, W = crop_bgr.shape[:2]
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 9
    )

    # Une ligeramente bordes del cuadro para que el contorno salga estable
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), iterations=1)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    side_min = int(min(H, W) * 0.030)  # un poco más sensible
    side_max = int(min(H, W) * 0.110)  # permite cuadros algo más grandes

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        ar = w / float(h + 1e-6)

        if ar < 0.80 or ar > 1.25:
            continue
        if w < side_min or h < side_min:
            continue
        if w > side_max or h > side_max:
            continue

        # opcional: evita falsos positivos enormes
        if (w * h) > (side_max * side_max * 2.0):
            continue

        boxes.append((x, y, w, h))

    # orden visual
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    return boxes

# =============================
#  2) Detectar si está marcado (tinta dentro del checkbox)
# =============================
def checkbox_esta_marcado(crop_bgr, box, tinta_thresh=0.10):
    """
    tinta_thresh más bajo = más sensible.
    """
    x, y, w, h = box
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    pad = max(2, int(min(w, h) * 0.20))  # interior
    x2, y2 = x + w, y + h
    xi1, yi1 = x + pad, y + pad
    xi2, yi2 = x2 - pad, y2 - pad
    if xi2 <= xi1 or yi2 <= yi1:
        return False, 0.0

    roi = gray[yi1:yi2, xi1:xi2]

    roi_th = cv2.adaptiveThreshold(
        roi, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21, 7
    )

    ink = (roi_th > 0).mean()  # 0..1
    return (ink >= tinta_thresh), float(ink)

# =============================
#  3) Asignar etiqueta a checkbox (texto OCR cercano a la derecha)
# =============================
def asignar_texto_a_checkboxes(crop_bgr, checkbox_boxes):
    ocr_items = ocr_easy_detail(crop_bgr)

    textos = []
    for bbox, txt, conf in ocr_items:
        txt = _norm_text(txt)
        if not txt:
            continue
        x, y, w, h = _bbox_to_xywh(bbox)
        textos.append({"x": x, "y": y, "w": w, "h": h, "txt": txt, "conf": conf})

    asignaciones = []
    for (x, y, w, h) in checkbox_boxes:
        cy = y + h / 2.0

        cand = []
        for t in textos:
            tyc = t["y"] + t["h"] / 2.0
            # misma fila aproximada
            if abs(tyc - cy) <= max(10, h * 1.2):
                # a la derecha del checkbox
                if t["x"] >= x + w + 6:
                    dx = t["x"] - (x + w)
                    dy = abs(tyc - cy)
                    cand.append((dx + 3 * dy, t))  # penaliza más el “salirse” de fila

        if cand:
            cand.sort(key=lambda z: z[0])
            etiqueta = cand[0][1]["txt"]
        else:
            etiqueta = ""

        asignaciones.append({"box": (x, y, w, h), "label": etiqueta})

    return asignaciones

# =============================
#  4) Resumen final
# =============================
def resumen_checks_panel(crop_bgr, tinta_thresh=0.10):
    checkbox_boxes = detectar_checkboxes(crop_bgr)

    estados = []
    for b in checkbox_boxes:
        marcado, ink = checkbox_esta_marcado(crop_bgr, b, tinta_thresh=tinta_thresh)
        estados.append({"box": b, "checked": marcado, "ink": ink})

    asign = asignar_texto_a_checkboxes(crop_bgr, checkbox_boxes)
    box_to_label = {a["box"]: a["label"] for a in asign}

    marcados = []
    for e in estados:
        if e["checked"]:
            label = box_to_label.get(e["box"], "")
            label = _norm_text(label) if label else "(sin etiqueta detectada)"
            if label not in marcados:
                marcados.append(label)

    return {
        "total_checkboxes": len(checkbox_boxes),
        "marcados": marcados,
        "debug_estados": estados,
        "debug_labels": asign
    }

def dibujar_debug(crop_bgr, data):
    vis = crop_bgr.copy()

    # dibuja checkboxes: verde=marcado rojo=no
    for i, e in enumerate(data["debug_estados"]):
        x, y, w, h = e["box"]
        color = (0, 255, 0) if e["checked"] else (0, 0, 255)
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        cv2.putText(vis, str(i), (x, y - 6 if y > 10 else y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # dibuja etiquetas OCR (azul) para ver qué leyó
    for a in data["debug_labels"]:
        x, y, w, h = a["box"]
        if a["label"]:
            cv2.putText(vis, a["label"][:18], (x + w + 6, y + int(h * 0.8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)

    return vis

# =============================
#  APP
# =============================
st.set_page_config(page_title="CIRCULACIÓN: checks -> resumen", layout="wide")
st.title("✅ CIRCULACIÓN: detectar checks marcados y resumir")

uploaded = st.file_uploader("Sube SOLO el recorte del cuadro de CIRCULACIÓN (JPG/PNG)", type=["png", "jpg", "jpeg"])
if not uploaded:
    st.info("👆 Sube el recorte para comenzar.")
    st.stop()

pil_img = Image.open(uploaded).convert("RGB")
img_rgb = np.array(pil_img)
crop_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

colA, colB = st.columns([1, 1])

with colA:
    st.subheader("Recorte (entrada)")
    st.image(img_rgb, use_container_width=True)

with colB:
    st.subheader("Parámetros")
    tinta = st.slider("Sensibilidad de marcado (tinta dentro del checkbox)", 0.05, 0.25, 0.10, 0.01)

    if st.button("Generar resumen"):
        data = resumen_checks_panel(crop_bgr, tinta_thresh=tinta)

        st.write(f"Checkboxes detectados: {data['total_checkboxes']}")

        if data["marcados"]:
            st.success("✅ Marcados detectados:")
            for m in data["marcados"]:
                st.write(f"- {m}")
        else:
            st.warning("No detecté checks marcados. Baja un poquito la sensibilidad (tinta).")

        st.markdown("---")
        st.subheader("Debug visual (verde=marcado, rojo=no)")
        vis = dibujar_debug(crop_bgr, data)
        st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), use_container_width=True)
