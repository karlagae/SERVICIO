# SERVICIO

import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image

# --- Función de OCR ---
def ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(th, lang="spa")
    return text.strip()

# --- Detectar cuadros ---
def detectar_cuadros(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

    horizontal = thresh.copy()
    hsize = int(horizontal.shape[1] / 25)
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (hsize, 1))
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)

    vertical = thresh.copy()
    vsize = int(vertical.shape[0] / 25)
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vsize))
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)

    mask = horizontal + vertical
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 250 and h > 80:  
            boxes.append((x, y, w, h))
    return boxes

# --- App Streamlit ---
st.title("Extractor interactivo por clic 🖱️📄")

uploaded = st.file_uploader("Sube tu imagen o PDF convertido a imagen", type=["png","jpg","jpeg"])

if uploaded:
    image = np.array(Image.open(uploaded))
    st.image(image, caption="Imagen cargada", use_container_width=True)

    boxes = detectar_cuadros(image)

    # Dibujamos los cuadros
    img_display = image.copy()
    for i, (x,y,w,h) in enumerate(boxes):
        cv2.rectangle(img_display, (x,y), (x+w,y+h), (255,0,0), 3)
        cv2.putText(img_display, f"{i}", (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,0,0),2)

    click = st.image(img_display, caption="Haz clic en un recuadro", use_container_width=True)

    # Streamlit tiene evento de posición del clic
    click_data = st.experimental_get_query_params()

    if "x" in click_data and "y" in click_data:
        cx = int(click_data["x"][0])
        cy = int(click_data["y"][0])

        # Determinar qué cuadro tocó
        selected = None
        for i,(x,y,w,h) in enumerate(boxes):
            if x <= cx <= x+w and y <= cy <= y+h:
                selected = i

        if selected is not None:
            st.success(f"Recuadro seleccionado: {selected}")

            x, y, w, h = boxes[selected]
            crop = image[y:y+h, x:x+w]

            st.image(crop, caption="Recuadro recortado")

            texto = ocr(crop)
            st.text_area("Texto extraído", texto, height=250)
