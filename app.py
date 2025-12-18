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
    """Aplica OCR (EasyOCR) a un recorte BGR y devuelve texto."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = reader.readtext(img_rgb, detail=0)  # solo texto
    return " ".join(results).strip()

# =============================
#  Autocorrector tipo Word (ortografía)
# =============================

@st.cache_resource
def get_spellchecker():
    return SpellChecker(language="es")

spell = get_spellchecker()

def autocorregir_texto(texto: str):
    """
    Autocorrige palabras en español (ortografía por diccionario).
    Devuelve: (texto_corregido, lista_cambios[(original, sugerido), ...])
    """
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
#  Detección de recuadros (SUBCUADROS EXACTOS)
# =============================

def detectar_subcuadros(img_bgr, min_w=180, min_h=80, ignore_small=60):
    """
    Detecta recuadros cerrados exactos del formulario
    (Comunicación, Valores y Creencias, Agudeza Visual, etc.)
    Devuelve lista de (x,y,w,h).
    """
    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Binarización robusta
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Cerrar huequitos para "cerrar" rectángulos
    th = cv2.morphologyEx(
        th,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=2
    )

    # Detectar líneas horizontales/verticales
    horizontal = th.copy()
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, W // 40), 1))
    horizontal = cv2.erode(horizontal, h_kernel, iterations=1)
    horizontal = cv2.dilate(horizontal, h_kernel, iterations=2)

    vertical = th.copy()
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, H // 40)))
    vertical = cv2.erode(vertical, v_kernel, iterations=1)
    vertical = cv2.dilate(vertical, v_kernel, iterations=2)

    mask = cv2.add(horizontal, vertical)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # ignora checkboxes / microcuadros
        if w < ignore_small and h < ignore_small:
            continue

        # ignora mega-recuadro que engloba TODO
        if w > 0.95 * W and h > 0.95 * H:
            continue

        # recuadros tipo panel
        if w >= min_w and h >= min_h:
            boxes.append((x, y, w, h))

    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    return boxes

# ✅ Alias para que NO truene si en alguna parte sigues usando detectar_cuadros
def detectar_cuadros(img_bgr):
    return detectar_subcuadros(img_bgr)

def dibujar_cuadros(img_bgr, boxes):
    """Dibuja los recuadros y su índice sobre la imagen."""
    vis = img_bgr.copy()
    for i, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.putText(
            vis, str(i), (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2
        )
    return vis

# =============================
#  App Streamlit
# =============================

st.set_page_config(page_title="Extractor por recuadros", layout="wide")
st.title("🖱️ Extrae texto eligiendo un recuadro")

st.write(
    "1. Sube una **imagen escaneada** (JPG/PNG).  \n"
    "2. Detectamos automáticamente los **subcuadros exactos** (rojo + número).  \n"
    "3. Elige el recuadro en la lista.  \n"
    "4. Verás el **texto OCR** y luego el **texto autocorregido**."
)

# Sidebar para ajustar detección
st.sidebar.header("⚙️ Ajustes de detección")
min_w = st.sidebar.slider("Ancho mínimo del recuadro", 80, 600, 180, 10)
min_h = st.sidebar.slider("Alto mínimo del recuadro", 40, 400, 80, 10)
ignore_small = st.sidebar.slider("Ignorar microcuadros (checkbox)", 20, 120, 60, 5)

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

# Detectar recuadros (subcuadros exactos)
boxes = detectar_subcuadros(img_bgr, min_w=min_w, min_h=min_h, ignore_small=ignore_small)

if not boxes:
    st.warning("No se detectaron recuadros con estos parámetros. Ajusta los sliders del sidebar.")
    st.stop()

# Dibujar recuadros
img_con_cuadros = dibujar_cuadros(img_bgr, boxes)

st.subheader("Imagen con recuadros detectados")
st.image(cv2.cvtColor(img_con_cuadros, cv2.COLOR_BGR2RGB), use_container_width=True)

# Selector de recuadro
indices = list(range(len(boxes)))
idx = st.selectbox(
    "Selecciona el número de recuadro que quieres extraer",
    indices,
    format_func=lambda i: f"Recuadro {i}"
)

x, y, w_box, h_box = boxes[idx]
crop = img_bgr[y:y + h_box, x:x + w_box]

col1, col2 = st.columns(2)

with col1:
    st.subheader(f"Recuadro {idx} recortado")
    st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), use_container_width=True)

with col2:
    st.subheader("Texto extraído (EasyOCR)")
    texto_ocr = ocr_easy(crop)
    st.text_area("1) OCR (crudo)", texto_ocr, height=180)

    st.markdown("---")

    usar_autocorrector = st.checkbox("Aplicar autocorrector (tipo Word)", value=True)

    if usar_autocorrector:
        texto_corregido, cambios = autocorregir_texto(texto_ocr)

        st.subheader("Texto autocorregido")
        st.text_area("2) Corregido", texto_corregido, height=180)

        if cambios:
            st.caption(f"Cambios detectados: {len(cambios)}")
            st.dataframe(
                {"Original": [c[0] for c in cambios],
                 "Sugerido": [c[1] for c in cambios]},
                use_container_width=True
            )
        else:
            st.success("No detecté palabras para corregir (o ya estaban bien).")
    else:
        st.info("Autocorrector desactivado.")
