import streamlit as st
import cv2
import numpy as np
from PIL import Image
import easyocr
import re
from spellchecker import SpellChecker
from difflib import SequenceMatcher, get_close_matches
import pandas as pd
from io import BytesIO
from datetime import datetime
import time

# ==========================================================
# 🔐 USUARIOS (SIMPLES)
# ==========================================================
USUARIOS = {
    "karla": {"password": "1234", "nombre": "Karla", "rol": "admin"},
    "usuario1": {"password": "abcd", "nombre": "Usuario 1", "rol": "usuario"}
}

# ==========================================================
# 🔐 SESIÓN
# ==========================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user" not in st.session_state:
    st.session_state.user = None

if "login_time" not in st.session_state:
    st.session_state.login_time = None

if "bitacora" not in st.session_state:
    st.session_state.bitacora = []

# ==========================================================
# 🔐 LOGIN
# ==========================================================
def login_view():
    st.title("🔐 Plataforma de Digitalización")
    st.caption("Ingrese sus credenciales para continuar")

    username = st.text_input("Usuario")
    password = st.text_input("Contraseña", type="password")

    if st.button("Entrar"):
        user = USUARIOS.get(username)

        if user and user["password"] == password:
            st.session_state.logged_in = True
            st.session_state.user = user
            st.session_state.login_time = datetime.now()

            st.session_state.bitacora.append({
                "usuario": username,
                "nombre": user["nombre"],
                "entrada": st.session_state.login_time,
                "salida": None,
                "duracion (seg)": None
            })

            st.success("Acceso correcto")
            st.rerun()
        else:
            st.error("Usuario o contraseña incorrectos")

# ==========================================================
# 🔐 LOGOUT
# ==========================================================
def logout():
    if st.session_state.login_time:
        salida = datetime.now()
        duracion = int((salida - st.session_state.login_time).total_seconds())

        for registro in reversed(st.session_state.bitacora):
            if registro["salida"] is None:
                registro["salida"] = salida
                registro["duracion (seg)"] = duracion
                break

    st.session_state.logged_in = False
    st.session_state.user = None
    st.session_state.login_time = None

    st.rerun()

# ==========================================================
# 🔐 BLOQUEO
# ==========================================================
if not st.session_state.logged_in:
    login_view()
    st.stop()

# ==========================================================
# 🔐 SIDEBAR
# ==========================================================
st.sidebar.success(f"👤 {st.session_state.user['nombre']}")
st.sidebar.write(f"Rol: {st.session_state.user['rol']}")

if st.sidebar.button("Cerrar sesión"):
    logout()

if st.session_state.user["rol"] == "admin":
    with st.sidebar.expander("📊 Bitácora de accesos"):
        df = pd.DataFrame(st.session_state.bitacora)
        st.dataframe(df, use_container_width=True)

# ==========================================================
# -----------------------------
# Word opcional
# -----------------------------
try:
    from docx import Document
    DOCX_AVAILABLE = True
except ModuleNotFoundError:
    DOCX_AVAILABLE = False

# ==========================================================
# OCR
# ==========================================================
@st.cache_resource
def get_ocr_reader():
    return easyocr.Reader(['es'], gpu=False)

reader = get_ocr_reader()

def ocr_easy(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = reader.readtext(img_rgb, detail=0)
    return " ".join(results).strip()

# ==========================================================
# AUTOCORRECTOR
# ==========================================================
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

# ==========================================================
# UI
# ==========================================================
st.set_page_config(page_title="Detector de cuadros + OCR", layout="wide")

if st.session_state.logged_in:
    st.title("🧾 Detector de cuadros + OCR + Autocorrector (TODOS los cuadros)")

uploaded = st.file_uploader("Sube una imagen", type=["png", "jpg", "jpeg"])

if uploaded:
    pil_img = Image.open(uploaded).convert("RGB")
    img = np.array(pil_img)

    st.image(img, use_container_width=True)

    if st.button("Extraer texto"):
        texto = ocr_easy(img)

        texto_final, cambios = autocorregir_texto(texto)

        st.text_area("OCR", texto_final, height=200)

        if cambios:
            st.write("Correcciones:")
            st.dataframe(pd.DataFrame(cambios, columns=["Original", "Sugerido"]))



pagina = st.sidebar.radio("Navegación", ["OCR", "Resultados"])
