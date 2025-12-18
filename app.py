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

def _contiene(a, b, margen=10):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return (
        ax <= bx + margen and ay <= by + margen and
        ax + aw >= bx + bw - margen and ay + ah >= by + bh - margen
    )

def deduplicar_boxes(boxes, iou_thr=0.65):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[2]*b[3], reverse=True)
    out = []
    for b in boxes:
        if all(_iou(b, o) < iou_thr for o in out):
            out.append(b)
    return sorted(out, key=lambda b: (b[1], b[0]))

def filtrar_bordes(boxes, W, H, margin_ratio=0.015):
    """
    Quita cajas que tocan los bordes del documento (mega-cuadro típico).
    """
    m = int(min(W, H) * margin_ratio)
    out = []
    for (x, y, w, h) in boxes:
        if x <= m or y <= m or (x+w) >= (W-m) or (y+h) >= (H-m):
            continue
        out.append((x, y, w, h))
    return out

def quitar_contenedores(boxes, W, H):
    """
    Quita contenedores GRANDES que engloban muchos cuadros.
    (Así NO se elimina un panel real sólo porque tiene checkboxes.)
    """
    if not boxes:
        return []
    out = []
    img_area = float(W * H)
    for b in boxes:
        x, y, w, h = b
        area_ratio = (w*h) / img_area
        hijos = [c for c in boxes if c != b and _contiene(b, c, margen=12)]

        # contenedor típico: grande + contiene varios
        if area_ratio >= 0.22 and len(hijos) >= 3:
            continue

        out.append(b)
    return out

# =============================
# Detección robusta de cuadros grandes (Cloud-friendly)
# =============================
def detectar_cuadros_grandes_auto(img_bgr: np.ndarray):
    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 9
    )

    # 1) Extrae líneas (dos escalas) para capturar cuadros de distintos tamaños
    rects_all = []
    configs = [
        (max(35, W//22), max(35, H//22)),  # normal
        (max(28, W//28), max(28, H//28)),  # más sensible (cuadros medianos)
    ]

    for h_len, v_len in configs:
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))

        horiz = cv2.morphologyEx(th, cv2.MORPH_OPEN, h_kernel, iterations=1)
        vert  = cv2.morphologyEx(th, cv2.MORPH_OPEN, v_kernel, iterations=1)
        mask = cv2.add(horiz, vert)

        # 2) CLOSE suave (clave para NO pegar toda la hoja)
        ck = int(min(W, H) * 0.010)  # ~1% del tamaño
        ck = max(7, min(13, ck))     # acota
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ck, ck))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)

        # 3) Contornos (externos) y filtro por tamaño
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_area = float(W * H)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area_ratio = (w*h) / img_area

            # mínimos más bajos para que agarre paneles como "Respiración"
            if area_ratio < 0.0025:  # 0.25%
                continue
            if area_ratio > 0.80:    # evita caja gigante
                continue

            # evita tiras raras
            if w < 0.16 * W:
                continue
            if h < 0.02 * H:
                continue

            # opcional: “rectángulo real” (4 lados) si se puede
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) != 4 and area_ratio < 0.02:
                continue

            rects_all.append((x, y, w, h))

    # Postproceso fuerte para evitar el desastre:
    rects = deduplicar_boxes(rects_all, iou_thr=0.70)

    # Quita cajas que tocan borde (mata el mega-cuadro)
    rects = filtrar_bordes(rects, W, H, margin_ratio=0.012)

    # Quita contenedores grandes que engloban muchos
    rects = quitar_contenedores(rects, W, H)

    # Dedup final
    rects = deduplicar_boxes(rects, iou_thr=0.60)
    return rects

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
st.set_page_config(page_title="Detector automático de cuadros + OCR", layout="wide")
st.title("🧾 Detector automático de cuadros grandes + OCR + Autocorrector (Cloud)")

uploaded = st.file_uploader("Sube una imagen JPG/PNG", type=["png", "jpg", "jpeg"])
if not uploaded:
    st.info("👆 Sube una imagen para comenzar.")
    st.stop()

pil_img = Image.open(uploaded).convert("RGB")
img_rgb = np.array(pil_img)
img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

boxes = detectar_cuadros_grandes_auto(img_bgr)

if not boxes:
    st.error("No pude detectar cuadros grandes automáticamente en esta imagen.")
    st.stop()

vis = dibujar_cuadros(img_bgr, boxes)
st.subheader(f"Imagen con cuadros detectados — detectados: {len(boxes)}")
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
