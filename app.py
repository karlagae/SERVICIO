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
def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx); y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw); y2 = min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0

def deduplicar_boxes(boxes, iou_thr=0.65):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[2]*b[3], reverse=True)
    out = []
    for b in boxes:
        if all(_iou(b, o) < iou_thr for o in out):
            out.append(b)
    return sorted(out, key=lambda b: (b[1], b[0]))

def filtrar_bordes(boxes, W, H, margin_ratio=0.012):
    m = int(min(W, H) * margin_ratio)
    out = []
    for (x, y, w, h) in boxes:
        if x <= m or y <= m or (x+w) >= (W-m) or (y+h) >= (H-m):
            continue
        out.append((x, y, w, h))
    return out

# =============================
# Detector genérico de rectángulos por líneas (para hoja y para recortes)
# =============================
def detectar_rects_por_lineas(
    img_bgr: np.ndarray,
    min_area_ratio: float,
    max_area_ratio: float,
    min_w_ratio: float,
    min_h_ratio: float,
    h_div: int,
    v_div: int,
    close_ratio: float,
    close_iter: int,
    border_filter: bool = True,
    border_margin_ratio: float = 0.012,
):
    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 9
    )

    h_len = max(20, W // h_div)
    v_len = max(20, H // v_div)

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))

    horiz = cv2.morphologyEx(th, cv2.MORPH_OPEN, h_kernel, iterations=1)
    vert  = cv2.morphologyEx(th, cv2.MORPH_OPEN, v_kernel, iterations=1)
    mask = cv2.add(horiz, vert)

    ck = int(min(W, H) * close_ratio)
    ck = max(5, min(15, ck))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ck, ck))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=close_iter)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_area = float(W * H)
    boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area_ratio = (w*h) / img_area

        if area_ratio < min_area_ratio:
            continue
        if area_ratio > max_area_ratio:
            continue
        if w < min_w_ratio * W:
            continue
        if h < min_h_ratio * H:
            continue

        boxes.append((x, y, w, h))

    boxes = deduplicar_boxes(boxes, iou_thr=0.70)

    if border_filter:
        boxes = filtrar_bordes(boxes, W, H, margin_ratio=border_margin_ratio)

    boxes = deduplicar_boxes(boxes, iou_thr=0.60)
    return boxes

# =============================
# 1) Cuadros GRANDES en hoja
# =============================
def detectar_cuadros_grandes(img_bgr: np.ndarray):
    # Más estricto (sólo paneles grandes)
    return detectar_rects_por_lineas(
        img_bgr,
        min_area_ratio=0.0025,  # 0.25%
        max_area_ratio=0.80,
        min_w_ratio=0.16,
        min_h_ratio=0.02,
        h_div=22,
        v_div=22,
        close_ratio=0.010,
        close_iter=1,
        border_filter=True,
        border_margin_ratio=0.012
    )

# =============================
# 2) Subcuadros DENTRO de cada cuadro grande
# =============================
def detectar_subcuadros_en_recorte(crop_bgr: np.ndarray, sensibilidad: float):
    """
    sensibilidad: 0.0 (conservador) -> 1.0 (más agresivo)
    """
    # Ajuste automático según sensibilidad
    # (entre más sensibilidad, más permite cuadros pequeños)
    min_area = 0.020 - (0.012 * sensibilidad)      # 2.0% -> 0.8% del recorte
    min_w    = 0.25  - (0.10  * sensibilidad)      # 25% -> 15% del ancho del recorte
    min_h    = 0.18  - (0.08  * sensibilidad)      # 18% -> 10% del alto del recorte

    boxes = detectar_rects_por_lineas(
        crop_bgr,
        min_area_ratio=max(0.006, min_area),
        max_area_ratio=0.95,     # evita que devuelva el recorte completo
        min_w_ratio=max(0.12, min_w),
        min_h_ratio=max(0.08, min_h),
        h_div=18,                # más sensible (líneas más cortas)
        v_div=18,
        close_ratio=0.009,
        close_iter=1,
        border_filter=False      # IMPORTANTE: dentro del recorte NO filtres por borde
    )

    # Quitar “caja igual al recorte” (duplicado típico)
    H, W = crop_bgr.shape[:2]
    out = []
    for (x, y, w, h) in boxes:
        area_ratio = (w*h) / float(W*H)
        if area_ratio > 0.85:
            continue
        out.append((x, y, w, h))

    return deduplicar_boxes(out, iou_thr=0.60)

# =============================
def dibujar_cuadros(img_bgr, items):
    vis = img_bgr.copy()
    for i, it in enumerate(items):
        x, y, w, h = it["bbox"]
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 0, 255), 3)
        cv2.putText(vis, f'{i}', (x, max(25, y-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return vis

# =============================
# App Streamlit
# =============================
st.set_page_config(page_title="Detector automático de cuadros + OCR", layout="wide")
st.title("🧾 Detector de cuadros (grandes + subcuadros) + OCR + Autocorrector")

uploaded = st.file_uploader("Sube una imagen JPG/PNG", type=["png", "jpg", "jpeg"])
if not uploaded:
    st.info("👆 Sube una imagen para comenzar.")
    st.stop()

pil_img = Image.open(uploaded).convert("RGB")
img_rgb = np.array(pil_img)
img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

st.sidebar.header("⚙️ Detección")
detectar_sub = st.sidebar.checkbox("Detectar subcuadros dentro de cada cuadro grande", value=True)
sens = st.sidebar.slider("Sensibilidad subcuadros", 0.0, 1.0, 0.70, 0.05)

# 1) grandes
grandes = detectar_cuadros_grandes(img_bgr)

items = []
for gi, g in enumerate(grandes):
    items.append({"tipo": "GRANDE", "label": f"Grande {gi}", "bbox": g})

# 2) subcuadros dentro de grandes
if detectar_sub:
    for gi, (gx, gy, gw, gh) in enumerate(grandes):
        crop = img_bgr[gy:gy+gh, gx:gx+gw]
        subs = detectar_subcuadros_en_recorte(crop, sensibilidad=sens)

        for si, (sx, sy, sw, sh) in enumerate(subs):
            # coord global
            items.append({
                "tipo": "SUB",
                "label": f"Grande {gi} → Sub {si}",
                "bbox": (gx + sx, gy + sy, sw, sh)
            })

# Limpieza final global
# (evita duplicados si un subcuadro coincide con el grande)
boxes_global = [it["bbox"] for it in items]
boxes_global = deduplicar_boxes(boxes_global, iou_thr=0.75)

# reconstruir items manteniendo labels lo mejor posible:
items2 = []
for b in boxes_global:
    # buscar el primero que matchee fuerte
    best = None
    best_iou = 0
    for it in items:
        i = _iou(b, it["bbox"])
        if i > best_iou:
            best_iou = i
            best = it
    if best is None:
        best = {"tipo": "?", "label": "Cuadro", "bbox": b}
    items2.append({"tipo": best["tipo"], "label": best["label"], "bbox": b})
items = sorted(items2, key=lambda it: (it["bbox"][1], it["bbox"][0]))

if not items:
    st.error("No pude detectar cuadros. Sube la sensibilidad o prueba otra imagen.")
    st.stop()

vis = dibujar_cuadros(img_bgr, items)
st.subheader(f"Cuadros detectados: {len(items)}")
st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), use_container_width=True)

opciones = list(range(len(items)))
idx = st.selectbox(
    "Selecciona un cuadro",
    opciones,
    format_func=lambda i: f"{i} — {items[i]['label']} ({items[i]['tipo']})"
)

x, y, w, h = items[idx]["bbox"]
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
            st.caption(f"Cambios detectados: {len(cambios)}")
            st.dataframe(
                {"Original": [c[0] for c in cambios],
                 "Sugerido": [c[1] for c in cambios]},
                use_container_width=True
            )
        else:
            st.info("No detecté palabras para corregir (o ya estaban bien).")
