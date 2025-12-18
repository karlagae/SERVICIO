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
# Utils
# =============================
def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx); y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw); y2 = min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0

def deduplicar_boxes(boxes, thr=0.75):
    boxes = sorted(boxes, key=lambda b: b[2]*b[3], reverse=True)
    out = []
    for b in boxes:
        if all(iou(b, o) < thr for o in out):
            out.append(b)
    return sorted(out, key=lambda b: (b[1], b[0]))

def contar_hijos(boxes, parent):
    px, py, pw, ph = parent
    c = 0
    for b in boxes:
        if b == parent:
            continue
        x, y, w, h = b
        if x >= px and y >= py and (x+w) <= (px+pw) and (y+h) <= (py+ph):
            c += 1
    return c

# =============================
# DETECTOR ROBUSTO: CUADROS GRANDES DEL FORMULARIO
# (Datos personales, Signos vitales, Respiración, Circulación, etc.)
# =============================
def detectar_cuadros_formulario(img_bgr: np.ndarray, sensibilidad: float = 0.85):
    H, W = img_bgr.shape[:2]
    img_area = float(H * W)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 1) binarización (líneas negras -> blanco en th)
    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        31, 12
    )

    # 2) extraer líneas (esto es clave)
    #    OJO: kernels GRANDES para quedarnos con líneas de tablas, no texto
    h_len = int(max(40, W * (0.10 - 0.05 * sensibilidad)))   # ~10% del ancho
    v_len = int(max(40, H * (0.08 - 0.04 * sensibilidad)))   # ~8% del alto
    h_len = max(30, min(h_len, 160))
    v_len = max(30, min(v_len, 160))

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))

    horiz = cv2.morphologyEx(th, cv2.MORPH_OPEN, h_kernel, iterations=1)
    vert  = cv2.morphologyEx(th, cv2.MORPH_OPEN, v_kernel, iterations=1)

    mask = cv2.add(horiz, vert)

    # 3) cerrar huequitos (pero sin pegar paneles distintos)
    k = int(3 + 4 * sensibilidad)  # 3..7
    k = max(3, min(7, k))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    # 4) contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 5) filtros para QUEDARNOS con cuadros GRANDES (no checkboxes)
    #    min_area baja con sensibilidad
    min_area_ratio = 0.004 - 0.002 * sensibilidad   # 0.4% -> 0.2%
    min_area_ratio = max(0.0015, min_area_ratio)

    # ancho/alto mínimos (relativos)
    min_w_ratio = 0.18 - 0.06 * sensibilidad        # 18% -> 12%
    min_h_ratio = 0.05 - 0.02 * sensibilidad        # 5%  -> 3%
    min_w_ratio = max(0.10, min_w_ratio)
    min_h_ratio = max(0.03, min_h_ratio)

    max_area_ratio = 0.70  # permite “Datos personales” completo, pero evita el marco total

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # evita mini cosas
        if w < 90 or h < 70:
            continue

        area_ratio = (w * h) / img_area
        if area_ratio < min_area_ratio:
            continue
        if area_ratio > max_area_ratio:
            continue

        if w < W * min_w_ratio:
            continue
        if h < H * min_h_ratio:
            continue

        boxes.append((x, y, w, h))

    # 6) deduplicar
    boxes = deduplicar_boxes(boxes, thr=0.78)

    # 7) quitar cuadros "padre" que engloban demasiado
    #    (ej. un cuadro enorme que contiene muchos)
    refined = []
    for b in boxes:
        hijos = contar_hijos(boxes, b)
        # si contiene muchísimos, es padre; lo quitamos
        if hijos >= 6:
            continue
        refined.append(b)

    refined = deduplicar_boxes(refined, thr=0.75)
    return refined

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
st.set_page_config(page_title="Detector de cuadros + OCR", layout="wide")
st.title("🧾 Detector de cuadros del formulario + OCR + Autocorrector")

uploaded = st.file_uploader("Sube una imagen JPG/PNG", type=["png", "jpg", "jpeg"])
if not uploaded:
    st.info("👆 Sube una imagen para comenzar.")
    st.stop()

pil_img = Image.open(uploaded).convert("RGB")
img_rgb = np.array(pil_img)
img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

st.sidebar.header("⚙️ Detección")
sens = st.sidebar.slider("Sensibilidad (más alto = detecta más cuadros)", 0.0, 1.0, 0.90, 0.05)

boxes = detectar_cuadros_formulario(img_bgr, sensibilidad=sens)

if not boxes:
    st.warning("No se detectaron cuadros. Sube la sensibilidad a 0.95–1.0.")
    st.stop()

vis = dibujar_cuadros(img_bgr, boxes)
st.subheader(f"Cuadros detectados: {len(boxes)}")
st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), use_container_width=True)

idx = st.selectbox("Selecciona un cuadro", list(range(len(boxes))), format_func=lambda i: f"Cuadro {i}")
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
