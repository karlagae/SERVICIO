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

def ocr_easy(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = reader.readtext(img_rgb, detail=0)
    return " ".join(results).strip()

# =============================
#  Autocorrector tipo Word
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
#  Helpers de boxes
# =============================
def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw*ah + bw*bh - inter
    return inter / union if union > 0 else 0.0

def deduplicar_boxes(boxes, thr=0.80):
    # Mantiene la caja más grande cuando se empalman mucho (evita 0=1, 2=3)
    boxes = sorted(boxes, key=lambda b: b[2]*b[3], reverse=True)
    out = []
    for b in boxes:
        if all(iou(b, o) < thr for o in out):
            out.append(b)
    return sorted(out, key=lambda b: (b[1], b[0]))

def dibujar_cuadros(img_bgr, boxes):
    vis = img_bgr.copy()
    for i, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 0, 255), 3)
        cv2.putText(vis, str(i), (x, max(20, y-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return vis

# =============================
#  Detector robusto por líneas (2 escalas)
# =============================
def detectar_rectangulos_por_lineas(img_bgr, scale="big",
                                   min_area_ratio=0.002,
                                   max_area_ratio=0.90,
                                   min_w_px=140,
                                   min_h_px=80):
    """
    Detecta rectángulos cerrados usando extracción de líneas horizontales + verticales.
    scale: "big" o "mid" (cambia tamaño de kernels).
    """
    H, W = img_bgr.shape[:2]
    img_area = float(H * W)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # binarización (líneas negras -> blanco)
    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 10
    )

    # kernels según escala
    if scale == "big":
        h_len = max(60, int(W * 0.12))
        v_len = max(60, int(H * 0.10))
        close_k = max(9, int(min(W, H) * 0.010))  # ~1%
    else:  # "mid"
        h_len = max(35, int(W * 0.06))
        v_len = max(35, int(H * 0.05))
        close_k = max(7, int(min(W, H) * 0.008))

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))

    horiz = cv2.morphologyEx(th, cv2.MORPH_OPEN, h_kernel, iterations=1)
    vert  = cv2.morphologyEx(th, cv2.MORPH_OPEN, v_kernel, iterations=1)

    mask = cv2.add(horiz, vert)

    # cerrar huecos para “cerrar” rectángulos
    k = int(close_k)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)),
        iterations=1
    )

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area_ratio = (w * h) / img_area

        if area_ratio < min_area_ratio:
            continue
        if area_ratio > max_area_ratio:
            continue
        if w < min_w_px or h < min_h_px:
            continue

        # evita el “mega cuadro” de toda la hoja si aparece
        if w > 0.97 * W and h > 0.97 * H:
            continue

        boxes.append((x, y, w, h))

    return boxes

def detectar_todos_los_cuadros(img_bgr,
                              min_area_pct=0.20,   # % área mínima
                              max_area_pct=90.0,   # % área máxima
                              min_w_px=140,
                              min_h_px=80):
    H, W = img_bgr.shape[:2]
    min_area_ratio = min_area_pct / 100.0
    max_area_ratio = max_area_pct / 100.0

    big = detectar_rectangulos_por_lineas(
        img_bgr, scale="big",
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio,
        min_w_px=min_w_px,
        min_h_px=min_h_px
    )
    mid = detectar_rectangulos_por_lineas(
        img_bgr, scale="mid",
        min_area_ratio=min_area_ratio * 0.45,     # más sensible
        max_area_ratio=max_area_ratio,
        min_w_px=max(110, int(min_w_px * 0.75)),
        min_h_px=max(70,  int(min_h_px * 0.85))
    )

    boxes = big + mid

    # dedup para que NO se repitan (0=1, 2=3)
    boxes = deduplicar_boxes(boxes, thr=0.78)

    return boxes

# =============================
#  App Streamlit
# =============================
st.set_page_config(page_title="Detector de cuadros + OCR + Autocorrector", layout="wide")
st.title("🧾 Detector de cuadros + OCR + Autocorrector")

uploaded = st.file_uploader("Sube una imagen JPG/PNG", type=["png", "jpg", "jpeg"])
if not uploaded:
    st.info("👆 Sube una imagen para comenzar.")
    st.stop()

pil_img = Image.open(uploaded).convert("RGB")
img_rgb = np.array(pil_img)
img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

st.sidebar.header("⚙️ Parámetros")

min_area_pct = st.sidebar.slider("Área mínima (%)", 0.05, 5.0, 0.20, 0.05)
max_area_pct = st.sidebar.slider("Área máxima (%)", 10.0, 95.0, 90.0, 1.0)

min_w_px = st.sidebar.slider("Ancho mínimo (px)", 80, 600, 140, 10)
min_h_px = st.sidebar.slider("Alto mínimo (px)", 50, 500, 80, 10)

boxes = detectar_todos_los_cuadros(
    img_bgr,
    min_area_pct=min_area_pct,
    max_area_pct=max_area_pct,
    min_w_px=min_w_px,
    min_h_px=min_h_px
)

if not boxes:
    st.warning("No se detectaron cuadros. Baja Área mínima o baja Ancho/Alto mínimo.")
    st.stop()

vis = dibujar_cuadros(img_bgr, boxes)
st.subheader(f"Imagen con cuadros detectados ({len(boxes)})")
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
            st.dataframe(
                {"Original": [c[0] for c in cambios],
                 "Sugerido": [c[1] for c in cambios]},
                use_container_width=True
            )
