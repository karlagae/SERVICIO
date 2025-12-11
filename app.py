import cv2
import numpy as np
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import easyocr

# =============================
#  OCR con EasyOCR (cacheado)
# =============================

@st.cache_resource
def get_ocr_reader():
    # Español (puedes agregar 'en' si quieres: ['es', 'en'])
    return easyocr.Reader(['es'], gpu=False)

reader = get_ocr_reader()

def ocr_easy(img_bgr):
    """
    Aplica OCR (EasyOCR) a un recorte BGR y devuelve texto.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = reader.readtext(img_rgb, detail=0)  # solo texto
    return " ".join(results).strip()

# =============================
#  Detección de recuadros
# =============================

def detectar_cuadros(img_bgr):
    """
    Detecta recuadros grandes en la imagen usando morfología.
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
    hsize = max(10, horizontal.shape[1] // 25)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hsize, 1))
    horizontal = cv2.erode(horizontal, h_kernel)
    horizontal = cv2.dilate(horizontal, h_kernel)

    # Líneas verticales
    vertical = thresh.copy()
    vsize = max(10, vertical.shape[0] // 25)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vsize))
    vertical = cv2.erode(vertical, v_kernel)
    vertical = cv2.dilate(vertical, v_kernel)

    mask = horizontal + vertical

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Filtro para cuadros grandes (ajusta a tu gusto)
        if w > 250 and h > 80:
            boxes.append((x, y, w, h))

    # Ordenar de arriba a abajo, izquierda a derecha
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    return boxes

def dibujar_cuadros(img_bgr, boxes):
    """
    Dibuja los recuadros y su índice sobre la imagen.
    """
    vis = img_bgr.copy()
    for i, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.putText(vis, str(i), (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return vis

def encontrar_cuadro_por_click(boxes, cx, cy):
    """
    Dado un punto (cx, cy) en coordenadas de la imagen,
    devuelve el índice del recuadro que contiene el punto
    o None si no cae en ninguno.
    """
    for i, (x, y, w, h) in enumerate(boxes):
        if x <= cx <= x + w and y <= cy <= y + h:
            return i
    return None

# =============================
#  App Streamlit
# =============================

st.set_page_config(page_title="Extractor interactivo por recuadros", layout="wide")
st.title("🖱️ Extrae texto dando clic en un recuadro")

st.write(
    "1. Sube una **imagen escaneada de la hoja** (JPG/PNG).  \n"
    "2. Se detectan automáticamente los recuadros (rojo + número).  \n"
    "3. Dibuja un rectángulo encima del recuadro que te interese (un solo clic y arrastras).  \n"
    "4. La app recorta ese recuadro y extrae el texto."
)

uploaded = st.file_uploader(
    "Sube una imagen (no PDF, mejor conviértelo a imagen antes)",
    type=["png", "jpg", "jpeg"]
)

if not uploaded:
    st.info("👆 Esperando que subas una imagen…")
    st.stop()

# Cargar imagen
pil_img = Image.open(uploaded).convert("RGB")
img_rgb = np.array(pil_img)
img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

# Detectar recuadros
boxes = detectar_cuadros(img_bgr)

if not boxes:
    st.warning("No se detectaron recuadros grandes. Revisa la calidad del escaneo.")
    st.stop()

img_con_cuadros = dibujar_cuadros(img_bgr, boxes)
h, w, _ = img_con_cuadros.shape

st.subheader("Imagen con recuadros detectados (haz un rectángulo sobre el que quieras)")
# Usamos st_canvas como lienzo interactivo
canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 0)",  # transparente
    stroke_width=2,
    stroke_color="#00FF00",
    background_image=Image.fromarray(cv2.cvtColor(img_con_cuadros, cv2.COLOR_BGR2RGB)),
    update_streamlit=True,
    height=h,
    width=w,
    drawing_mode="rect",  # dibujar rectángulos
    key="canvas",
)

# Determinar el último rectángulo dibujado (si lo hay)
selected_index = None
if canvas_result.json_data is not None:
    objects = canvas_result.json_data.get("objects", [])
    if len(objects) > 0:
        last_obj = objects[-1]  # tomamos el último rect dibujado
        # Coordenadas del rectángulo en el canvas
        left = last_obj["left"]
        top = last_obj["top"]
        rect_w = last_obj["width"]
        rect_h = last_obj["height"]

        # Centro del rectángulo que dibujó el usuario
        cx = left + rect_w / 2
        cy = top + rect_h / 2

        selected_index = encontrar_cuadro_por_click(boxes, cx, cy)

if selected_index is None:
    st.info("Dibuja un rectángulo sobre un recuadro para verlo y hacer OCR.")
    st.stop()

st.success(f"Recuadro detectado: {selected_index}")

x, y, w_box, h_box = boxes[selected_index]
crop = img_bgr[y:y + h_box, x:x + w_box]

col1, col2 = st.columns(2)

with col1:
    st.subheader(f"Recuadro {selected_index} recortado")
    st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), use_container_width=True)

with col2:
    st.subheader("Texto extraído (EasyOCR)")
    texto = ocr_easy(crop)
    st.text_area("Resultado OCR", texto, height=300)
