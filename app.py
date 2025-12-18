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
    # Español (puedes agregar 'en' si quieres)
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

    # Palabras con acentos/ñ
    patron = r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+"
    tokens = re.findall(patron, texto)

    # desconocidas (en minúsculas)
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
#  Detección de recuadros
# =============================

def detectar_cuadros(img_bgr):
    """
    Detector híbrido:
    A) Rectángulos cerrados (cajas)
    B) Bloques de líneas horizontales (formularios)
    Devuelve bounding boxes (x, y, w, h) sin duplicados.
    """
    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # ---------- A) RECTÁNGULOS CERRADOS ----------
    threshA = cv2.adaptiveThreshold(
        ~gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        15, -2
    )

    horizontalA = threshA.copy()
    hsize = max(10, W // 25)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hsize, 1))
    horizontalA = cv2.erode(horizontalA, h_kernel)
    horizontalA = cv2.dilate(horizontalA, h_kernel)

    verticalA = threshA.copy()
    vsize = max(10, H // 25)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vsize))
    verticalA = cv2.erode(verticalA, v_kernel)
    verticalA = cv2.dilate(verticalA, v_kernel)

    maskA = horizontalA + verticalA
    contoursA, _ = cv2.findContours(maskA, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxesA = []
    for cnt in contoursA:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 250 and h > 80:
            boxesA.append((x, y, w, h))

    # ---------- B) BLOQUES DE LÍNEAS (OBS/NOTA) ----------
    _, threshB = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, W // 15), 1))
    horizontalB = cv2.morphologyEx(threshB, cv2.MORPH_OPEN, kernel_h)

    horizontalB = cv2.dilate(
        horizontalB,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15)),
        iterations=2
    )

    contoursB, _ = cv2.findContours(horizontalB, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxesB = []
    for cnt in contoursB:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > W * 0.4 and h > 60:
            boxesB.append((x, y, w, h))

    # ---------- UNIR + QUITAR DUPLICADOS ----------
    def iou(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x1 = max(ax, bx)
        y1 = max(ay, by)
        x2 = min(ax + aw, bx + bw)
        y2 = min(ay + ah, by + bh)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        areaA = aw * ah
        areaB = bw * bh
        union = areaA + areaB - inter
        return inter / union if union > 0 else 0

    all_boxes = boxesA + boxesB
    final = []

    for b in sorted(all_boxes, key=lambda x: (x[1], x[0])):
        dup = False
        for f in final:
            if iou(b, f) > 0.5:   # si se traslapan mucho, es el mismo bloque
                dup = True
                break
        if not dup:
            final.append(b)

    return final


def dibujar_cuadros(img_bgr, boxes):
    """
    Dibuja los recuadros y su índice sobre la imagen.
    """
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
    "2. Detectamos automáticamente los recuadros (rojo + número).  \n"
    "3. Elige el recuadro en la lista.  \n"
    "4. Verás el **texto OCR** y luego el **texto autocorregido** (tipo Word)."
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
    st.warning("No se detectaron recuadros grandes. Revisa la calidad del escaneo o ajusta el filtro.")
    st.stop()

# Dibujar recuadros
img_con_cuadros = dibujar_cuadros(img_bgr, boxes)

st.subheader("Imagen con recuadros detectados")
st.image(cv2.cvtColor(img_con_cuadros, cv2.COLOR_BGR2RGB),
         use_container_width=True)

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
    st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
             use_container_width=True)

with col2:
    # OCR
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
