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

# -----------------------------
# Word opcional (no debe tronar)
# -----------------------------
try:
    from docx import Document
    DOCX_AVAILABLE = True
except ModuleNotFoundError:
    DOCX_AVAILABLE = False


# ==========================================================
#  OCR con EasyOCR (cacheado)
# ==========================================================
@st.cache_resource
def get_ocr_reader():
    return easyocr.Reader(['es'], gpu=False)

reader = get_ocr_reader()

def ocr_easy(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = reader.readtext(img_rgb, detail=0)
    return " ".join(results).strip()

def ocr_easy_detail(img_bgr):
    """
    Devuelve lista: [ (bbox, text, conf), ... ]
    bbox = [[x,y],[x,y],[x,y],[x,y]]
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return reader.readtext(img_rgb, detail=1)

# ==========================================================
#  Autocorrector (diccionario tipo Word)
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

# ==========================================================
#  Utilidades detección (IOU / containment / dedupe)
# ==========================================================
def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw*ah + bw*bh - inter
    return inter / union if union > 0 else 0

def contains(big, small, margin=2):
    bx, by, bw, bh = big
    sx, sy, sw, sh = small
    return (sx >= bx + margin and sy >= by + margin and
            sx + sw <= bx + bw - margin and sy + sh <= by + bh - margin)

def dedupe_boxes(boxes, iou_thr=0.85):
    boxes = sorted(boxes, key=lambda b: (b[2]*b[3]), reverse=True)  # grandes primero
    out = []
    for b in boxes:
        if all(iou(b, o) < iou_thr for o in out):
            out.append(b)
    # orden visual
    out = sorted(out, key=lambda b: (b[1], b[0]))
    return out

# ==========================================================
#  Detección robusta de CUADROS/PANELES
# ==========================================================
def detectar_cuadros_formulario(
    img_bgr,
    min_area_ratio=0.002,
    max_area_ratio=0.45,
    min_w_ratio=0.08,
    min_h_ratio=0.03,
    rectangularidad_min=0.55,
    close_kernel=5,
    close_iter=1,
    iou_thr=0.85,
    remover_contenedores=True,
    contenedor_min_hijos=3,
    contenedor_factor_area=1.8
):
    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 9
    )

    k = max(3, int(close_kernel))
    th = cv2.morphologyEx(
        th, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)),
        iterations=max(1, int(close_iter))
    )

    contours, hierarchy = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    img_area = float(H * W)
    candidatos = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area <= 0:
            continue

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

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        rect_fill = area / float(w * h)
        if rect_fill < rectangularidad_min:
            continue

        if len(approx) >= 4 and cv2.isContourConvex(approx):
            candidatos.append((x, y, w, h))
        else:
            if area_ratio >= (min_area_ratio * 2.0):
                candidatos.append((x, y, w, h))

    candidatos = dedupe_boxes(candidatos, iou_thr=iou_thr)

    if remover_contenedores and len(candidatos) > 1:
        areas = [b[2]*b[3] for b in candidatos]
        hijos_count = []
        for i, b in enumerate(candidatos):
            c = 0
            for j, s in enumerate(candidatos):
                if i != j and contains(b, s, margin=3):
                    c += 1
            hijos_count.append(c)

        filtrados = []
        for b, c in zip(candidatos, hijos_count):
            b_area = b[2]*b[3]
            if c >= contenedor_min_hijos:
                med = np.median(areas) if len(areas) else b_area
                if b_area > med * contenedor_factor_area:
                    continue
            filtrados.append(b)

        candidatos = dedupe_boxes(filtrados, iou_thr=iou_thr)

    return candidatos

def dibujar_cuadros(img_bgr, boxes):
    vis = img_bgr.copy()
    for i, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 0, 255), 3)
        cv2.putText(
            vis, str(i), (x, max(25, y-10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2
        )
    return vis


# ==========================================================
#  Checkboxes (tu lógica, intacta)
# ==========================================================
EXPECTED_CIRCULACION = [
    "Sin compromiso", "Comprometida", "Especifique", "Cianosis", "Hormigueo", "Extremidades frías",
    "Edema",
    "N", "Negativo",
    "1+ Leve", "2+ Moderado", "3+ Severo",
    "Llenado capilar",
    "Menor a 3 seg", "Mayor a 3 seg",
    "Mucosas",
    "Hidratadas", "Deshidratadas",
    "Condiciones de la piel",
    "Rosada", "Pálida", "Cianótica", "Ictérica", "Marmórea"
]

def _preprocess_for_boxes(crop_bgr):
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 9
    )
    th = cv2.medianBlur(th, 3)
    return th

def detectar_checkboxes(crop_bgr):
    th = _preprocess_for_boxes(crop_bgr)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    H, W = th.shape[:2]
    boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if w < 12 or h < 12:
            continue
        if w > int(W * 0.12) or h > int(H * 0.12):
            continue

        ar = w / float(h)
        if ar < 0.7 or ar > 1.3:
            continue

        area = cv2.contourArea(cnt)
        rect_area = w * h
        if rect_area <= 0:
            continue
        if area / rect_area < 0.15:
            continue

        pad = max(1, int(min(w, h) * 0.18))
        x0 = max(0, x + pad); y0 = max(0, y + pad)
        x1 = min(W, x + w - pad); y1 = min(H, y + h - pad)
        if x1 <= x0 or y1 <= y0:
            continue

        inner = th[y0:y1, x0:x1]
        fill = float(np.mean(inner > 0))  # 0..1

        boxes.append({"x": x, "y": y, "w": w, "h": h, "fill": fill})

    boxes = sorted(boxes, key=lambda b: (b["y"], b["x"]))
    return boxes

def _ocr_rect(bbox):
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return min(xs), min(ys), max(xs), max(ys)

def _clean_text(t: str) -> str:
    t = re.sub(r"\s+", " ", t).strip()
    t = t.replace("|", "").replace("’", "'")
    return t

def _best_match_expected(label: str, expected_list, min_ratio=0.62):
    if not label:
        return None
    lab = _clean_text(label)

    candidates = get_close_matches(lab, expected_list, n=1, cutoff=min_ratio)
    if candidates:
        return candidates[0]

    best = None
    best_r = 0.0
    for e in expected_list:
        r = SequenceMatcher(None, lab.lower(), e.lower()).ratio()
        if r > best_r:
            best_r = r
            best = e
    return best if best_r >= min_ratio else lab

def asignar_label_checkbox_bidir(crop_bgr, checkbox, ocr_results, dx_ratio=0.02, dy_ratio=0.05):
    H, W = crop_bgr.shape[:2]
    x = checkbox["x"]; y = checkbox["y"]; w = checkbox["w"]; h = checkbox["h"]
    cb_cx = x + w/2.0
    cb_cy = y + h/2.0

    y_min = max(0, int(cb_cy - H * dy_ratio))
    y_max = min(H, int(cb_cy + H * dy_ratio))

    left_x_min  = max(0, int(cb_cx - W * 0.55))
    left_x_max  = max(0, int(cb_cx - W * dx_ratio))
    right_x_min = min(W, int(cb_cx + W * dx_ratio))
    right_x_max = min(W, int(cb_cx + W * 0.55))

    left_candidates = []
    right_candidates = []

    for (bbox, text, conf) in ocr_results:
        tx1, ty1, tx2, ty2 = _ocr_rect(bbox)
        cx = (tx1 + tx2) / 2.0
        cy = (ty1 + ty2) / 2.0

        if cy < y_min or cy > y_max:
            continue

        t = _clean_text(text)
        if not t:
            continue

        if cx >= left_x_min and cx <= left_x_max:
            dx = cb_cx - cx
            dy = abs(cb_cy - cy)
            score = dx*dx + (dy*dy * 0.6)
            left_candidates.append((score, cx, t))

        if cx >= right_x_min and cx <= right_x_max:
            dx = cx - cb_cx
            dy = abs(cb_cy - cy)
            score = dx*dx + (dy*dy * 0.6)
            right_candidates.append((score, cx, t))

    left_candidates.sort(key=lambda z: z[0])
    right_candidates.sort(key=lambda z: z[0])

    if left_candidates:
        pool = [c for c in left_candidates[:6]]
    elif right_candidates:
        pool = [c for c in right_candidates[:6]]
    else:
        return None

    pool_sorted = sorted(pool, key=lambda z: z[1])
    words = [p[2] for p in pool_sorted]

    label = " ".join(words)
    label = _clean_text(label)
    label = re.sub(r"\b(\w+)\s+\1\b", r"\1", label, flags=re.IGNORECASE)
    label = _best_match_expected(label, EXPECTED_CIRCULACION, min_ratio=0.62)
    return label

def _grupo_por_x(crop_bgr, checkbox):
    H, W = crop_bgr.shape[:2]
    cx = checkbox["x"] + checkbox["w"]/2.0
    x_ratio = cx / float(W)

    if x_ratio < 0.28:
        return "Estado"
    elif x_ratio < 0.43:
        return "Edema"
    elif x_ratio < 0.62:
        return "Llenado capilar"
    elif x_ratio < 0.78:
        return "Mucosas"
    else:
        return "Condiciones de la piel"

def resumen_checkboxes(crop_bgr, umbral_marcado=0.10, debug=False):
    ocr_det = ocr_easy_detail(crop_bgr)
    checks = detectar_checkboxes(crop_bgr)

    detalle = []
    marcados = []

    for ch in checks:
        label = asignar_label_checkbox_bidir(crop_bgr, ch, ocr_det)
        marcado = (ch["fill"] >= umbral_marcado)
        grupo = _grupo_por_x(crop_bgr, ch)

        item = {
            "grupo": grupo,
            "label": label if label else "(sin texto detectado)",
            "marcado": marcado,
            "fill": round(ch["fill"], 3),
            "x": ch["x"], "y": ch["y"], "w": ch["w"], "h": ch["h"]
        }
        detalle.append(item)
        if marcado and label:
            marcados.append((grupo, label))

    resumen = {}
    for g, lab in marcados:
        resumen.setdefault(g, [])
        if lab not in resumen[g]:
            resumen[g].append(lab)

    img_debug = None
    if debug:
        img_debug = crop_bgr.copy()
        for it in detalle:
            x,y,w,h = it["x"], it["y"], it["w"], it["h"]
            color = (0,255,0) if it["marcado"] else (0,0,255)
            cv2.rectangle(img_debug, (x,y), (x+w,y+h), color, 2)
            cv2.putText(img_debug, f'{it["fill"]:.2f}', (x, max(12, y-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    return resumen, detalle, img_debug


# ==========================================================
#  Exportadores
# ==========================================================
def build_excel_bytes(df_all, df_cambios, df_resumen_cb, df_detalle_cb):
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df_all.to_excel(writer, index=False, sheet_name="OCR_TODOS")
        df_cambios.to_excel(writer, index=False, sheet_name="Cambios_Autocorrect")
        df_resumen_cb.to_excel(writer, index=False, sheet_name="Resumen_Checkboxes")
        df_detalle_cb.to_excel(writer, index=False, sheet_name="Detalle_Checkboxes")
    bio.seek(0)
    return bio.getvalue()

def build_word_bytes(df_all, df_resumen_cb):
    doc = Document()
    doc.add_heading("Resultados de extracción (OCR)", level=1)
    doc.add_paragraph(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    doc.add_heading("OCR por cuadro", level=2)
    for _, r in df_all.iterrows():
        doc.add_heading(f"Cuadro {int(r['Cuadro'])}", level=3)
        doc.add_paragraph("OCR (crudo):")
        doc.add_paragraph(str(r.get("OCR_Crudo", "")))
        doc.add_paragraph("OCR (final):")
        doc.add_paragraph(str(r.get("OCR_Final", "")))
        doc.add_paragraph("")

    doc.add_heading("Resumen de checkboxes", level=2)
    if df_resumen_cb.empty:
        doc.add_paragraph("Sin resumen de checkboxes (no se activó o no se detectó).")
    else:
        for _, r in df_resumen_cb.iterrows():
            doc.add_paragraph(f"Cuadro {int(r['Cuadro'])} | {r['Seccion']}: {r['Marcados']}")

    bio = BytesIO()
    doc.save(bio)
    bio.seek(0)
    return bio.getvalue()


# ==========================================================
#  App Streamlit
# ==========================================================
st.set_page_config(page_title="Detector de cuadros + OCR", layout="wide")
st.title("🧾 Detector de cuadros + OCR + Autocorrector (TODOS los cuadros)")

uploaded = st.file_uploader("Sube una imagen JPG/PNG", type=["png", "jpg", "jpeg"])
if not uploaded:
    st.info("👆 Sube una imagen para comenzar.")
    st.stop()

pil_img = Image.open(uploaded).convert("RGB")
img_rgb = np.array(pil_img)
img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

st.sidebar.header("⚙️ Detección (ajustable)")
min_area_pct = st.sidebar.slider("Área mínima (%)", 0.05, 5.00, 0.20, 0.05)
max_area_pct = st.sidebar.slider("Área máxima (%)", 5.00, 80.00, 45.00, 1.00)
min_w_pct = st.sidebar.slider("Ancho mínimo (% del ancho)", 4, 60, 8, 1)
min_h_pct = st.sidebar.slider("Alto mínimo (% del alto)", 2, 40, 3, 1)
rect_fill = st.sidebar.slider("Rectangularidad mínima (relleno)", 0.30, 0.95, 0.55, 0.05)
close_kernel = st.sidebar.slider("Close kernel", 3, 21, 5, 2)
close_iter = st.sidebar.slider("Close iteraciones", 1, 4, 1, 1)
iou_thr = st.sidebar.slider("Quitar duplicados (IOU)", 0.60, 0.95, 0.85, 0.01)

st.sidebar.markdown("---")
remover_cont = st.sidebar.checkbox("Quitar cuadros contenedores (mega-cuadro)", value=True)
cont_hijos = st.sidebar.slider("Contenedor: mínimo #hijos", 2, 10, 3, 1)
cont_factor = st.sidebar.slider("Contenedor: factor de área", 1.1, 3.0, 1.8, 0.1)

st.sidebar.markdown("---")
aplicar_autocorrect = st.sidebar.checkbox("Aplicar autocorrector a TODOS", value=True)
hacer_checkboxes = st.sidebar.checkbox("Intentar resumen de checkboxes en TODOS", value=True)
umbral_cb = st.sidebar.slider("Umbral checkboxes (tinta)", 0.03, 0.50, 0.10, 0.01)
mostrar_debug_cb = st.sidebar.checkbox("Mostrar debug de checkboxes", value=False)

boxes = detectar_cuadros_formulario(
    img_bgr,
    min_area_ratio=min_area_pct/100.0,
    max_area_ratio=max_area_pct/100.0,
    min_w_ratio=min_w_pct/100.0,
    min_h_ratio=min_h_pct/100.0,
    rectangularidad_min=rect_fill,
    close_kernel=close_kernel,
    close_iter=close_iter,
    iou_thr=iou_thr,
    remover_contenedores=remover_cont,
    contenedor_min_hijos=cont_hijos,
    contenedor_factor_area=cont_factor
)

if not boxes:
    st.warning("No se detectaron cuadros con estos parámetros. Ajusta sliders en el sidebar.")
    st.stop()

vis = dibujar_cuadros(img_bgr, boxes)
st.subheader("Imagen con cuadros detectados")
st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), use_container_width=True)

st.markdown("---")
st.subheader("▶️ Extraer TODO (Cuadro 0, 1, 2, …)")

btn = st.button("🚀 Extraer todos los cuadros")
if not btn:
    st.info("Presiona el botón para extraer todos los cuadros y habilitar la descarga del Excel/Word.")
    st.stop()

progress = st.progress(0)
status = st.empty()

rows_all = []
rows_cambios = []
rows_resumen_cb = []
rows_detalle_cb = []

for i, (x, y, w, h) in enumerate(boxes):
    status.write(f"Procesando Cuadro {i}...")
    crop = img_bgr[y:y+h, x:x+w]

    # OCR
    texto_ocr = ocr_easy(crop)
    texto_final = texto_ocr
    cambios_final = []

    if aplicar_autocorrect:
        texto_ok, cambios = autocorregir_texto(texto_ocr)
        texto_final = texto_ok
        cambios_final = cambios or []

    rows_all.append({
        "Cuadro": i,
        "x": x, "y": y, "w": w, "h": h,
        "OCR_Crudo": texto_ocr,
        "OCR_Final": texto_final
    })

    for (orig, sug) in cambios_final:
        rows_cambios.append({
            "Cuadro": i,
            "Original": orig,
            "Sugerido": sug
        })

    # Checkboxes (opcional, puede no aplicar a todos los cuadros)
    if hacer_checkboxes:
        try:
            resumen, detalle, img_debug = resumen_checkboxes(crop, umbral_marcado=umbral_cb, debug=mostrar_debug_cb)
        except Exception:
            resumen, detalle, img_debug = {}, [], None

        if isinstance(resumen, dict) and resumen:
            for k, vals in resumen.items():
                rows_resumen_cb.append({
                    "Cuadro": i,
                    "Seccion": k,
                    "Marcados": ", ".join(vals)
                })

        if isinstance(detalle, list) and detalle:
            for it in detalle:
                rows_detalle_cb.append({"Cuadro": i, **it})

    # Mostrar resultados en pantalla (como pediste: cuadro 0, abajo cuadro 1, etc.)
    with st.expander(f"Cuadro {i} — ver resultado", expanded=(i == 0)):
        st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.text_area(f"Cuadro {i} | OCR (crudo)", texto_ocr, height=140)
        st.text_area(f"Cuadro {i} | OCR (final)", texto_final, height=140)

        if aplicar_autocorrect and cambios_final:
            st.caption(f"Cambios autocorrector: {len(cambios_final)}")
            st.dataframe(pd.DataFrame(cambios_final, columns=["Original", "Sugerido"]), use_container_width=True)

        if hacer_checkboxes:
            if rows_resumen_cb:
                # mostramos solo lo del cuadro actual
                df_tmp = pd.DataFrame([r for r in rows_resumen_cb if r["Cuadro"] == i])
                if not df_tmp.empty:
                    st.caption("Resumen checkboxes (si aplica)")
                    st.dataframe(df_tmp[["Seccion", "Marcados"]], use_container_width=True)

    progress.progress(int(((i+1) / len(boxes)) * 100))

status.success("✅ Listo. Ya puedes descargar el archivo.")

df_all = pd.DataFrame(rows_all)
df_cambios = pd.DataFrame(rows_cambios) if rows_cambios else pd.DataFrame(columns=["Cuadro", "Original", "Sugerido"])
df_resumen_cb = pd.DataFrame(rows_resumen_cb) if rows_resumen_cb else pd.DataFrame(columns=["Cuadro", "Seccion", "Marcados"])
df_detalle_cb = pd.DataFrame(rows_detalle_cb) if rows_detalle_cb else pd.DataFrame(columns=["Cuadro", "grupo", "label", "marcado", "fill", "x", "y", "w", "h"])

st.markdown("---")
st.subheader("📦 Descargar resultados")

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
base = f"extraccion_todos_{ts}"

excel_bytes = build_excel_bytes(df_all, df_cambios, df_resumen_cb, df_detalle_cb)
st.download_button(
    "⬇️ Descargar Excel (TODO)",
    data=excel_bytes,
    file_name=f"{base}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

if DOCX_AVAILABLE:
    docx_bytes = build_word_bytes(df_all, df_resumen_cb)
    st.download_button(
        "⬇️ Descargar Word (TODO)",
        data=docx_bytes,
        file_name=f"{base}.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
else:
    st.info("Word desactivado: falta instalar 'python-docx'. (Excel sí funciona).")
