import cv2
import numpy as np
import pytesseract
from PIL import Image
import streamlit as st

# =============================
#  Funciones auxiliares
# =============================

def ocr(img_bgr):
    """
    Aplica OCR (Tesseract) a un recorte BGR (OpenCV) y devuelve texto.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(th, lang="spa")
    return text.strip()


def detectar_cuadros(img_bgr):
    """
    Detecta recuadros grandes en la imagen usando líneas rectas.
    Devuelve lista de bounding boxes (x, y, w, h).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        ~gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        15, -2
    )

    # Líneas horizontales
    horizontal = thresh.copy()
    hsize = horizontal.shape[1] // 25
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hsize, 1))
    horizontal = cv2.erode(horizontal, h_kernel)
    horizontal = cv2.dilate(horizontal, h_kernel)

    # Líneas verticales
    vertical = thresh.copy()
    vsize = vertical.shape[0] // 25
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vsize))
    vertical = cv2.erode(vertical, v_kernel)
    vertical = cv2.dilate(vertical, v_kernel)

    mask = horizontal + vertical

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # FILTRO para evitar recuadros muy pequeños
        if w > 250 and h > 80:
            boxes.append((x, y, w, h))

    # Ordenar de arriba a abajo, izquierda a derecha
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    return boxes


def dibujar_cuadros(img_bgr, boxes):
    """
    Dibuja los recuadros y un índice (0, 1, 2, ...) sobre la imagen.
    """
    vis = img_bgr.copy()
    for i, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.putText(vis, str(i), (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return vis

# =============================
#  App Streamlit
# =============================

st.set_page_config(page_title="Extractor por recuadros", layout="wide")

st.title("🖱️ Extractor de texto por recuadros")
st.write(
    "Sube una **imagen del formulario** (JPG/PNG). "
    "El sistema detecta los recuadros, los numera, "
    "y tú eliges cuál quieres extraer."
)

uploaded = st.file_uploader(
    "Sube una imagen (escaneo de la hoja, con recuadros)",
    type=["png", "jpg", "jpeg"]
)

if uploaded is None:
    st.info("👆 Esperando que subas una imagen...")
    st.stop()

# Cargar imagen con PIL y convertir a BGR (OpenCV)
pil_img = Image.open(uploaded).convert("RGB")
img_rgb = np.array(pil_img)
img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

# Detectar recuadros
boxes = detectar_cuadros(img_bgr)

if not boxes:
    st.warning("No se detectaron recuadros grandes. "
               "Quizá debas ajustar el escaneo o los filtros.")
    st.stop()

# Mostrar imagen con recuadros numerados
img_con_cuadros = dibujar_cuadros(img_bgr, boxes)
st.subheader("Imagen con recuadros detectados")
st.image(cv2.cvtColor(img_con_cuadros, cv2.COLOR_BGR2RGB),
         use_container_width=True)

# Selector para elegir recuadro
indices = list(range(len(boxes)))
idx = st.selectbox(
    "Selecciona el número de recuadro que quieres extraer",
    indices,
    format_func=lambda i: f"Recuadro {i}"
)

x, y, w, h = boxes[idx]
crop = img_bgr[y:y + h, x:x + w]

col1, col2 = st.columns(2)

with col1:
    st.subheader(f"Recuadro {idx} recortado")
    st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
             use_container_width=True)

with col2:
    st.subheader("Texto extraído (OCR)")
    texto = ocr(crop)
    st.text_area("Resultado OCR", texto, height=300)
