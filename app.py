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
#  Helpers de boxes (NUEVO)
# =============================
def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw*ah + bw*bh - inter
    return inter / union if union > 0 else 0.0

def _deduplicar_boxes(boxes, thr=0.80):
    # deja el más grande cuando se empalman fuerte
    boxes = sorted(boxes, key=lambda b: b[2]*b[3], reverse=True)
    out = []
    for b in boxes:
        if all(_iou(b, o) < thr for o in out):
            out.append(b)
    return sorted(out, key=lambda b: (b[1], b[0]))

def _quitar_padres_que_engloban(boxes):
    # elimina cajas "padre" que contienen muchas otras
    out = []
    for b in boxes:
        bx, by, bw, bh = b
        hijos = 0
        for c in boxes:
            if c == b:
                continue
            x, y, w, h = c
            inside = (x >= bx and y >= by and (x+w) <= (bx+bw) and (y+h) <= (by+bh))
            if inside and (w*h) < 0.92*(bw*bh):
                hijos += 1
        if hijos >= 6:   # si engloba muchas, es padre -> fuera
            continue
        out.append(b)
    return out


# =============================
#  TU detector principal (lo dejo igual pero con nombre claro)
#  (si ya lo tienes diferente, puedes reemplazar SOLO esta función)
# =============================
def detectar_cuadros_grandes(
    img_bgr,
    min_area_ratio=0.010,
    max_area_ratio=0.80,
    min_w_ratio=0.12,
    min_h_ratio=0.03,
    close_kernel=11,
    close_iter=1
):
    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 9
    )

    # líneas horizontales/verticales
    h_len = max(30, W // 25)
    v_len = max(30, H // 25)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))

    horiz = cv2.morphologyEx(th, cv2.MORPH_OPEN, h_kernel, iterations=1)
    vert  = cv2.morphologyEx(th, cv2.MORPH_OPEN, v_kernel, iterations=1)

    mask = cv2.add(horiz, vert)

    k = max(3, int(close_kernel))
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)),
        iterations=max(1, int(close_iter))
    )

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    img_area = float(H * W)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area_ratio = (w*h) / img_area

        if area_ratio < min_area_ratio:
            continue
        if area_ratio > max_area_ratio:
            continue
        if w < W * min_w_ratio:
            continue
        if h < H * min_h_ratio:
            continue

        boxes.append((x, y, w, h))

    boxes = _deduplicar_boxes(boxes, thr=0.85)
    return boxes


# =============================
#  REFUERZO: detector MEDIANO/PEQUEÑO (NUEVO)
#  (este es el que te saca Respiración y Valoración)
# =============================
def detectar_cuadros_medianos(img_bgr):
    H, W = img_bgr.shape[:2]
    img_area = float(H * W)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        31, 12
    )

    # kernels más chicos
    h_len = max(25, int(W * 0.045))
    v_len = max(25, int(H * 0.040))

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))

    horiz = cv2.morphologyEx(th, cv2.MORPH_OPEN, h_kernel, iterations=1)
    vert  = cv2.morphologyEx(th, cv2.MORPH_OPEN, v_kernel, iterations=1)

    mask = cv2.add(horiz, vert)

    # cerrar un poquito más para líneas rotas
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)),
        iterations=1
    )

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        ar = (w*h) / img_area

        # filtros para medianos
        if w < 140 or h < 90:
            continue
        if ar < 0.0012 or ar > 0.35:
            continue
        if w < W * 0.10 or h < H * 0.035:
            continue

        boxes.append((x, y, w, h))

    boxes = _deduplicar_boxes(boxes, thr=0.85)
    return boxes


# =============================
#  FUSIÓN FINAL (NUEVO)
# =============================
def detectar_todos_los_cuadros(img_bgr, params_grandes):
    grandes = detectar_cuadros_grandes(img_bgr, **params_grandes)
    medianos = detectar_cuadros_medianos(img_bgr)

    # unir
    boxes = grandes + medianos

    # quitar duplicados (arregla 0 y 1 iguales)
    boxes = _deduplicar_boxes(boxes, thr=0.78)

    # quitar padres (arregla cuadro 4 que junta 3)
    boxes = _quitar_padres_que_engloban(boxes)
    boxes = _deduplicar_boxes(boxes, thr=0.78)

    return boxes


def dibujar_cuadros(img_bgr, boxes):
    vis = img_bgr.copy()
    for i, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 0, 255), 3)
        cv2.putText(vis, str(i), (x, max(20, y-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return vis


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

st.sidebar.header("⚙️ Parámetros (ajustables)")

min_area_ratio = st.sidebar.slider("Área mínima (%)", 0.1, 5.0, 1.0, 0.1) / 100.0
max_area_ratio = st.sidebar.slider("Área máxima (%)", 10.0, 90.0, 80.0, 1.0) / 100.0
min_w_ratio = st.sidebar.slider("Ancho mínimo (% ancho)", 5, 60, 12, 1) / 100.0
min_h_ratio = st.sidebar.slider("Alto mínimo (% alto)", 2, 40, 3, 1) / 100.0
close_kernel = st.sidebar.slider("Close kernel", 3, 25, 11, 2)
close_iter = st.sidebar.slider("Close iteraciones", 1, 3, 1, 1)

params_grandes = dict(
    min_area_ratio=min_area_ratio,
    max_area_ratio=max_area_ratio,
    min_w_ratio=min_w_ratio,
    min_h_ratio=min_h_ratio,
    close_kernel=close_kernel,
    close_iter=close_iter
)

# ✅ aquí usamos: tu detector + refuerzo + merge
boxes = detectar_todos_los_cuadros(img_bgr, params_grandes)

if not boxes:
    st.warning("No se detectaron cuadros. Ajusta los sliders.")
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
