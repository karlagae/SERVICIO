import cv2
import numpy as np
from PIL import Image
import streamlit as st
import easyocr
import re
from spellchecker import SpellChecker

# =============================
# OCR (EasyOCR)
# =============================
@st.cache_resource
def get_ocr_reader():
    return easyocr.Reader(["es"], gpu=False)

reader = get_ocr_reader()

def ocr_easy(img_bgr: np.ndarray) -> str:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    parts = reader.readtext(img_rgb, detail=0)
    return " ".join(parts).strip()

# =============================
# Autocorrector (diccionario)
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
# Utils geométricos
# =============================
def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx); y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw); y2 = min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0

def deduplicar_boxes(boxes, thr=0.70):
    # conserva el más grande cuando hay solapes fuertes
    boxes = sorted(boxes, key=lambda b: b[2]*b[3], reverse=True)
    out = []
    for b in boxes:
        if all(iou(b, o) < thr for o in out):
            out.append(b)
    return sorted(out, key=lambda b: (b[1], b[0]))

# =============================
# Detector: SUBCUADROS EN TODA LA HOJA (lo que tú pides)
# =============================
def detectar_subcuadros_hoja(img_bgr: np.ndarray, sensibilidad: float = 0.70):
    """
    Detecta paneles/rectángulos grandes del formulario (Respiración, Circulación, etc.)
    directamente en TODA la hoja.

    sensibilidad:
      0.0 -> más conservador (menos cuadros)
      1.0 -> más agresivo (más cuadros)
    """
    H, W = img_bgr.shape[:2]
    img_area = float(H * W)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # 1) binarización
    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 9
    )

    # 2) bordes
    edges = cv2.Canny(th, 50, 150)

    # 3) cerrar huecos pequeños (OJO: kernel chico para NO juntar paneles)
    #    kernel sube un poquito con sensibilidad, pero con límite
    k = int(3 + 4 * sensibilidad)   # 3..7
    k = max(3, min(7, k))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 4) contornos (TREE para encontrar cuadros internos)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # ====== filtros dinámicos (por tamaño relativo)
    # área mínima: baja con sensibilidad
    min_area_ratio = 0.006 - 0.003 * sensibilidad   # 0.6% -> 0.3%
    min_area_ratio = max(0.0025, min_area_ratio)

    # área máxima: evita mega-cuadros
    max_area_ratio = 0.40

    # ancho/alto mínimos: bajan con sensibilidad
    min_w_ratio = 0.22 - 0.08 * sensibilidad        # 22% -> 14%
    min_h_ratio = 0.06 - 0.02 * sensibilidad        # 6% -> 4%
    min_w_ratio = max(0.10, min_w_ratio)
    min_h_ratio = max(0.03, min_h_ratio)

    # evita checkboxes
    min_abs_w = int(60)
    min_abs_h = int(60)

    boxes = []

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        if peri < 50:
            continue

        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)  # aproximación a polígono

        # Queremos rectángulos (4 lados) y convexos
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        x, y, w, h = cv2.boundingRect(approx)

        # quita checkboxes
        if w < min_abs_w and h < min_abs_h:
            continue

        area_ratio = (w * h) / img_area

        # quita mega-cuadro
        if area_ratio > max_area_ratio:
            continue

        # muy pequeño
        if area_ratio < min_area_ratio:
            continue

        # filtros por proporción
        if w < W * min_w_ratio:
            continue
        if h < H * min_h_ratio:
            continue

        # quita “marco exterior” pegado a bordes
        margin = int(0.01 * min(W, H))
        if x <= margin or y <= margin or (x + w) >= (W - margin) or (y + h) >= (H - margin):
            continue

        boxes.append((x, y, w, h))

    # quitar duplicados por solape
    boxes = deduplicar_boxes(boxes, thr=0.72)

    # extra: eliminar cajas que contienen a otras muchas (típico cuadro padre)
    # (conserva las más “útiles”)
    final = []
    for b in boxes:
        bx, by, bw, bh = b
        contains = 0
        for c in boxes:
            if c == b:
                continue
            cx, cy, cw, ch = c
            if cx >= bx and cy >= by and (cx+cw) <= (bx+bw) and (cy+ch) <= (by+bh):
                contains += 1
        # si contiene demasiadas, es “padre” y lo quitamos
        if contains >= 4:
            continue
        final.append(b)

    final = deduplicar_boxes(final, thr=0.70)
    return final

# =============================
def dibujar_cuadros(img_bgr, boxes):
    vis = img_bgr.copy()
    for i, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 0, 255), 3)
        cv2.putText(vis, str(i), (x, max(25, y-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return vis

# =============================
# App Streamlit
# =============================
st.set_page_config(page_title="Subcuadros + OCR", layout="wide")
st.title("🧾 Subcuadros automáticos (toda la hoja) + OCR + Autocorrector")

uploaded = st.file_uploader("Sube una imagen JPG/PNG", type=["png", "jpg", "jpeg"])
if not uploaded:
    st.info("👆 Sube una imagen para comenzar.")
    st.stop()

pil_img = Image.open(uploaded).convert("RGB")
img_rgb = np.array(pil_img)
img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

st.sidebar.header("⚙️ Detección")
sens = st.sidebar.slider("Sensibilidad (más alto = detecta más)", 0.0, 1.0, 0.85, 0.05)

boxes = detectar_subcuadros_hoja(img_bgr, sensibilidad=sens)

if not boxes:
    st.warning("No se detectaron subcuadros. Sube la sensibilidad a 0.95–1.0.")
    st.stop()

vis = dibujar_cuadros(img_bgr, boxes)
st.subheader(f"Subcuadros detectados: {len(boxes)}")
st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), use_container_width=True)

idx = st.selectbox("Selecciona un subcuadro", list(range(len(boxes))), format_func=lambda i: f"Subcuadro {i}")
x, y, w, h = boxes[idx]
crop = img_bgr[y:y+h, x:x+w]

col1, col2 = st.columns(2)

with col1:
    st.subheader("Recorte")
    st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), use_container_width=True)

with col2:
    st.subheader("OCR + Autocorrector")
    texto_ocr = ocr_easy(crop)
    st.text_area("OCR (crudo)", texto_ocr, height=180)

    if st.checkbox("Aplicar autocorrector", value=True):
        texto_ok, cambios = autocorregir_texto(texto_ocr)
        st.text_area("Corregido", texto_ok, height=180)
        if cambios:
            st.caption(f"Cambios: {len(cambios)}")
            st.dataframe(
                {"Original": [c[0] for c in cambios],
                 "Sugerido": [c[1] for c in cambios]},
                use_container_width=True
            )
