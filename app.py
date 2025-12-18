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
#  Detección: CUADROS GRANDES (hoja completa)
# =============================
def detectar_cuadros_grandes(
    img_bgr,
    min_area_ratio=0.010,
    max_area_ratio=0.30,
    min_w_ratio=0.18,
    min_h_ratio=0.06,
    close_kernel=7,
    close_iter=2
):
    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 9
    )

    h_len = max(30, W // 25)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    horiz = cv2.morphologyEx(th, cv2.MORPH_OPEN, h_kernel, iterations=1)

    v_len = max(30, H // 25)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    vert = cv2.morphologyEx(th, cv2.MORPH_OPEN, v_kernel, iterations=1)

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

    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    return boxes

# =============================
#  Split: dividir cuadros pegados
# =============================
def _split_por_valles_verticales(crop_bin, min_gap_px=18):
    col_sum = crop_bin.sum(axis=0) / 255.0
    if col_sum.max() > 0:
        col_sum = col_sum / col_sum.max()

    valle = col_sum < 0.08

    cuts = []
    start = None
    for i, v in enumerate(valle):
        if v and start is None:
            start = i
        if (not v) and start is not None:
            end = i - 1
            if (end - start) >= min_gap_px:
                cuts.append((start + end) // 2)
            start = None
    if start is not None:
        end = len(valle) - 1
        if (end - start) >= min_gap_px:
            cuts.append((start + end) // 2)

    return cuts

def dividir_cuadros_pegados(img_bgr, boxes, split_w_ratio=0.55, min_gap_px=18):
    H, W = img_bgr.shape[:2]
    out = []

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 9
    )

    for (x, y, w, h) in boxes:
        if w < W * split_w_ratio:
            out.append((x, y, w, h))
            continue

        crop = th[y:y + h, x:x + w]
        cuts = _split_por_valles_verticales(crop, min_gap_px=min_gap_px)

        if len(cuts) == 0:
            out.append((x, y, w, h))
            continue

        xs = [0] + cuts + [w]
        for i in range(len(xs) - 1):
            x0, x1 = xs[i], xs[i + 1]
            seg_w = x1 - x0
            if seg_w < 0.12 * W:
                continue
            out.append((x + x0, y, seg_w, h))

    out = sorted(out, key=lambda b: (b[1], b[0]))
    return out

# =============================
#  Deduplicar: elimina cuadros repetidos
# =============================
def deduplicar_boxes(boxes, iou_thr=0.65):
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)

    def iou(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x1 = max(ax, bx)
        y1 = max(ay, by)
        x2 = min(ax + aw, bx + bw)
        y2 = min(ay + ah, by + bh)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0

    final = []
    for b in boxes:
        if all(iou(b, f) < iou_thr for f in final):
            final.append(b)

    final = sorted(final, key=lambda b: (b[1], b[0]))
    return final

# =============================
#  Subcuadros (opcional)
# =============================
def detectar_subcuadros(img_bgr):
    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                          iterations=1)
    contours, _ = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if (w < 50 and h < 50):
            continue
        if (w > 0.95 * W and h > 0.95 * H):
            continue
        if w > 160 and h > 70:
            boxes.append((x, y, w, h))

    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    return boxes

def dibujar_cuadros(img_bgr, boxes):
    vis = img_bgr.copy()
    for i, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.putText(vis, str(i), (x, max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return vis

# =============================
#  App Streamlit
# =============================
st.set_page_config(page_title="Extractor por cuadros", layout="wide")
st.title("🧾 Detector de cuadros + OCR + Autocorrector")

uploaded = st.file_uploader("Sube una imagen JPG/PNG", type=["png", "jpg", "jpeg"])
if not uploaded:
    st.info("👆 Sube una imagen para comenzar.")
    st.stop()

pil_img = Image.open(uploaded).convert("RGB")
img_rgb = np.array(pil_img)
img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

st.sidebar.header("⚙️ Detección")
modo = st.sidebar.radio("Modo", ["Cuadros grandes (hoja completa)", "Subcuadros (paneles internos)"])

if modo == "Cuadros grandes (hoja completa)":
    min_area_ratio = st.sidebar.slider("Área mínima (%)", 0.1, 5.0, 1.0, 0.1) / 100.0
    max_area_ratio = st.sidebar.slider("Área máxima (%)", 5.0, 60.0, 30.0, 1.0) / 100.0
    min_w_ratio = st.sidebar.slider("Ancho mínimo (% del ancho)", 5, 60, 18, 1) / 100.0
    min_h_ratio = st.sidebar.slider("Alto mínimo (% del alto)", 2, 40, 6, 1) / 100.0
    close_kernel = st.sidebar.slider("Close kernel (fCierre)", 3, 25, 7, 2)
    close_iter = st.sidebar.slider("Close iteraciones", 1, 4, 2, 1)

    boxes = detectar_cuadros_grandes(
        img_bgr,
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio,
        min_w_ratio=min_w_ratio,
        min_h_ratio=min_h_ratio,
        close_kernel=close_kernel,
        close_iter=close_iter
    )

    activar_split = st.sidebar.checkbox("Dividir cuadros pegados (split automático)", value=True)
    split_w_ratio = st.sidebar.slider("Split: umbral ancho (% ancho)", 35, 90, 55, 1) / 100.0
    min_gap_px = st.sidebar.slider("Split: separación mínima (px)", 8, 60, 18, 1)

    if activar_split and boxes:
        boxes = dividir_cuadros_pegados(img_bgr, boxes, split_w_ratio=split_w_ratio, min_gap_px=min_gap_px)

    # ✅ elimina duplicados (arregla Cuadro 0 y Cuadro 1 iguales)
    iou_thr = st.sidebar.slider("Deduplicar: IoU umbral", 0.40, 0.90, 0.65, 0.01)
    boxes = deduplicar_boxes(boxes, iou_thr=iou_thr)

else:
    boxes = detectar_subcuadros(img_bgr)

if not boxes:
    st.warning("No se detectaron cuadros con estos parámetros. Ajusta sliders en el sidebar.")
    st.stop()

vis = dibujar_cuadros(img_bgr, boxes)
st.subheader("Imagen con cuadros detectados")
st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), use_container_width=True)

idx = st.selectbox("Selecciona un cuadro", list(range(len(boxes))), format_func=lambda i: f"Cuadro {i}")
x, y, w, h = boxes[idx]
crop = img_bgr[y:y + h, x:x + w]

col1, col2 = st.columns(2)
with col1:
    st.subheader("Recorte")
    st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), use_container_width=True)

with col2:
    st.subheader("OCR + Autocorrector")
    texto_ocr = ocr_easy(crop)
    st.text_area("OCR (crudo)", texto_ocr, height=160)

    if st.checkbox("Aplicar autocorrector", value=True):
        texto_ok, cambios = autocorregir_texto(texto_ocr)
        st.text_area("Corregido", texto_ok, height=160)
        if cambios:
            st.dataframe(
                {"Original": [c[0] for c in cambios], "Sugerido": [c[1] for c in cambios]},
                use_container_width=True
            )
