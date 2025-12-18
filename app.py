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

            # Respeta capitalización
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
# Geometría / filtros
# =============================
def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx); y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw); y2 = min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0

def _contiene(a, b, margen=6):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return (
        ax <= bx + margen and ay <= by + margen and
        ax + aw >= bx + bw - margen and ay + ah >= by + bh - margen
    )

def _rectangularidad(contour, w, h):
    """Qué tan 'rectángulo lleno' es el contorno vs su bounding box."""
    area_c = abs(cv2.contourArea(contour))
    area_r = float(w * h) if w > 0 and h > 0 else 1.0
    return area_c / area_r

# =============================
# Detección AUTOMÁTICA de cuadros grandes (Cloud-friendly)
#   - Construye máscara de líneas
#   - Busca contornos con jerarquía (RETR_TREE)
#   - Se queda con rectángulos reales (approx 4 vértices)
#   - Quita "contenedores" que engloban varios cuadros
# =============================
def detectar_cuadros_grandes_automatico(img_bgr: np.ndarray):
    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Suaviza un poco para estabilizar threshold
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Binarización (líneas negras -> blanco)
    th = cv2.adaptiveThreshold(
        gray_blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 9
    )

    # Máscara de líneas (horizontal + vertical)
    h_len = max(40, W // 18)
    v_len = max(40, H // 18)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))

    horiz = cv2.morphologyEx(th, cv2.MORPH_OPEN, h_kernel, iterations=1)
    vert  = cv2.morphologyEx(th, cv2.MORPH_OPEN, v_kernel, iterations=1)
    mask = cv2.add(horiz, vert)

    # Cerrar pequeños huecos para que los rectángulos queden "cerrados"
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)

    # Contornos con jerarquía (IMPORTANTÍSIMO para evitar duplicados)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []

    img_area = float(H * W)

    candidatos = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area_ratio = (w * h) / img_area

        # Filtra por tamaño (cuadros GRANDES)
        if area_ratio < 0.006:   # 0.6% (baja para adobe scan)
            continue
        if area_ratio > 0.85:    # evita mega-cuadro de la hoja
            continue

        # evita “tiras” verticales inventadas
        if w < 0.18 * W:
            continue
        if h < 0.03 * H:
            continue

        # Aproxima a polígono y exige 4 lados
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4:
            continue

        # Rectangularidad: el contorno debe parecerse a su bbox
        rect_score = _rectangularidad(cnt, w, h)
        if rect_score < 0.25:
            continue

        candidatos.append((x, y, w, h))

    # Orden visual
    candidatos = sorted(candidatos, key=lambda b: (b[1], b[0]))

    # Deduplicado suave por IoU (elimina bordes casi iguales)
    dedup = []
    for b in candidatos:
        if all(_iou(b, d) < 0.70 for d in dedup):
            dedup.append(b)

    # Quitar "contenedores": si una caja contiene a 2+ cajas, se elimina
    # (esto mata el cuadro que engloba 3 secciones)
    finales = []
    for b in dedup:
        hijos = [c for c in dedup if c != b and _contiene(b, c, margen=10)]
        if len(hijos) >= 2:
            # b es contenedor -> descartar
            continue
        finales.append(b)

    # Segundo deduplicado por seguridad
    finales2 = []
    for b in sorted(finales, key=lambda x: x[2]*x[3], reverse=True):
        if all(_iou(b, f) < 0.55 for f in finales2):
            finales2.append(b)

    finales2 = sorted(finales2, key=lambda b: (b[1], b[0]))
    return finales2

def dibujar_cuadros(img_bgr, boxes):
    vis = img_bgr.copy()
    for i, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.putText(vis, str(i), (x, max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return vis

# =============================
# App Streamlit
# =============================
st.set_page_config(page_title="Detector de cuadros + OCR", layout="wide")
st.title("🧾 Detector automático de cuadros grandes + OCR + Autocorrector")

uploaded = st.file_uploader("Sube una imagen JPG/PNG", type=["png", "jpg", "jpeg"])
if not uploaded:
    st.info("👆 Sube una imagen para comenzar.")
    st.stop()

pil_img = Image.open(uploaded).convert("RGB")
img_rgb = np.array(pil_img)
img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

# Detecta AUTOMÁTICO (sin sliders)
boxes = detectar_cuadros_grandes_automatico(img_bgr)

if not boxes:
    st.error("No pude detectar cuadros grandes automáticamente en esta imagen.")
    st.stop()

vis = dibujar_cuadros(img_bgr, boxes)
st.subheader("Imagen con cuadros detectados (automático)")
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
