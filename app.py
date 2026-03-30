import streamlit as st
import numpy as np
from PIL import Image
import io
from datetime import date
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, white
from reportlab.lib.utils import ImageReader
from scipy.ndimage import gaussian_filter

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HEG Report Generator",
    page_icon="🧠",
    layout="centered"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.app-header {
    background: linear-gradient(135deg, #1a237e 0%, #0d47a1 50%, #01579b 100%);
    padding: 2rem; border-radius: 16px; text-align: center;
    margin-bottom: 1.5rem; box-shadow: 0 4px 20px rgba(13,71,161,0.3);
}
.app-header h1 { color: white; font-size: 1.7rem; font-weight: 700; margin: 0; }
.app-header p  { color: #bbdefb; font-size: 0.85rem; margin: 0.3rem 0 0; }

.section-card {
    background: white; border-radius: 12px; padding: 1.4rem;
    margin-bottom: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    border-left: 4px solid #1565c0;
}
.section-title {
    font-size: 0.8rem; font-weight: 700; color: #1565c0;
    text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.9rem;
}
.upload-label { font-size: 0.8rem; font-weight: 600; color: #1565c0; margin-bottom: 0.3rem; }

.stButton > button {
    background: linear-gradient(135deg, #1565c0, #0d47a1) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    font-size: 1rem !important; font-weight: 600 !important; width: 100% !important;
    box-shadow: 0 4px 12px rgba(21,101,192,0.4) !important; padding: 0.7rem !important;
}
.success-box {
    background: #e8f5e9; border: 1px solid #4caf50; border-radius: 10px;
    padding: 1rem; text-align: center; color: #2e7d32; font-weight: 600; margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1>🧠 Haemo-Encephalography Report</h1>
    <p>Dr. Hany Elhennawy Psychiatric Center &nbsp;·&nbsp; Wijdan Therapy</p>
</div>
""", unsafe_allow_html=True)

# ── Color mapping ─────────────────────────────────────────────────────────────
def apply_heg_colormap(img: Image.Image) -> Image.Image:
    gray = np.array(img.convert("L")).astype(np.float32) / 255.0
    gray = gaussian_filter(gray, sigma=8)
    gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)

    r = np.power(np.clip((gray - 0.5) * 2.0,       0, 1), 0.7)
    g = np.power(np.clip(1.0 - np.abs(gray - 0.5) * 2.5, 0, 1), 0.8)
    b = np.power(np.clip((0.45 - gray) * 2.5,      0, 1), 0.7)

    rgb = np.stack([r, g, b], axis=-1)
    return Image.fromarray((rgb * 255).astype(np.uint8), "RGB")

def pil_to_reportlab(img: Image.Image) -> ImageReader:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return ImageReader(buf)

# ── Patient Info ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-card"><div class="section-title">📋 Patient Information</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    patient_name = st.text_input("Patient Name", placeholder="e.g. Rema Menna")
    assistant1   = st.text_input("Assistant 1",  placeholder="e.g. Nawal")
with col2:
    report_date  = st.date_input("Date", value=date.today())
    assistant2   = st.text_input("Assistant 2",  placeholder="optional")
st.markdown('</div>', unsafe_allow_html=True)

# ── Enhancement ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-card"><div class="section-title">⚡ Enhancement Used</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1: eye_dev  = st.checkbox("Eye Deviation")
with c2: photic   = st.checkbox("Photic Stimulation")
with c3: bioptron = st.checkbox("Bioptron")
st.markdown('</div>', unsafe_allow_html=True)

# ── Image Upload ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-card"><div class="section-title">📸 Upload Forehead Photos</div>', unsafe_allow_html=True)
st.caption("Upload the 3 photos taken with the USB microscope. The app applies the HEG color map automatically.")

col_mf, col_lf, col_rf = st.columns(3)
with col_mf:
    st.markdown('<div class="upload-label">Mid Frontal</div>', unsafe_allow_html=True)
    mid_file   = st.file_uploader("mid",   type=["jpg","jpeg","png"], label_visibility="collapsed", key="mid")
with col_lf:
    st.markdown('<div class="upload-label">Left Frontal</div>', unsafe_allow_html=True)
    left_file  = st.file_uploader("left",  type=["jpg","jpeg","png"], label_visibility="collapsed", key="left")
with col_rf:
    st.markdown('<div class="upload-label">Right Frontal</div>', unsafe_allow_html=True)
    right_file = st.file_uploader("right", type=["jpg","jpeg","png"], label_visibility="collapsed", key="right")
st.markdown('</div>', unsafe_allow_html=True)

# ── Process & Preview ─────────────────────────────────────────────────────────
files  = {"mid": mid_file, "left": left_file, "right": right_file}
labels = {"mid": "Mid Frontal", "left": "Left Frontal", "right": "Right Frontal"}
processed = {}

if any(f is not None for f in files.values()):
    st.markdown('<div class="section-card"><div class="section-title">🎨 HEG Color Preview</div>', unsafe_allow_html=True)
    prev_cols = st.columns(3)
    for i, (key, f) in enumerate(files.items()):
        with prev_cols[i]:
            st.caption(labels[key])
            if f is not None:
                raw     = Image.open(f).convert("RGB")
                colored = apply_heg_colormap(raw)
                processed[key] = colored
                st.image(colored, use_container_width=True)
            else:
                st.info("Not uploaded")
    st.markdown('</div>', unsafe_allow_html=True)

# ── Impression ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-card"><div class="section-title">📝 Doctor\'s Impression</div>', unsafe_allow_html=True)
impression = st.text_area(
    "Impression", height=100, label_visibility="collapsed",
    placeholder="Dr. Hany writes clinical impression here after reviewing the images..."
)
st.markdown('</div>', unsafe_allow_html=True)

# ── PDF Generator ─────────────────────────────────────────────────────────────
def generate_pdf(name, rdate, asst1, asst2, eye, pho, bio, imgs, imp):
    buf = io.BytesIO()
    W, H = A4
    M = 45
    cv = canvas.Canvas(buf, pagesize=A4)

    # Background
    cv.setFillColor(white); cv.rect(0, 0, W, H, fill=1, stroke=0)

    # Header
    cv.setFillColor(HexColor("#0d47a1")); cv.rect(0, H-58, W, 58, fill=1, stroke=0)
    cv.setFillColor(HexColor("#00acc1")); cv.rect(0, H-61, W, 3,  fill=1, stroke=0)
    cv.setFillColor(white); cv.setFont("Helvetica-Bold", 16)
    cv.drawCentredString(W/2, H-28, "Haemo-Encephalography Report")
    cv.setFont("Helvetica", 8); cv.setFillColor(HexColor("#bbdefb"))
    cv.drawCentredString(W/2, H-44, "Dr. Hany Elhennawy Psychiatric Center  ·  Wijdan Therapy")

    # Patient info
    iy = H - 85
    cv.setFillColor(HexColor("#e3f2fd"))
    cv.roundRect(M, iy-50, W-2*M, 54, 6, fill=1, stroke=0)

    def field(cx, cy, lbl, val):
        cv.setFillColor(HexColor("#1565c0")); cv.setFont("Helvetica-Bold", 9)
        cv.drawString(cx, cy, lbl)
        cv.setFillColor(HexColor("#212121")); cv.setFont("Helvetica", 9)
        cv.drawString(cx + cv.stringWidth(lbl, "Helvetica-Bold", 9) + 5, cy, val or "—")

    field(M+10,  iy-14, "Name:",        name)
    field(M+10,  iy-30, "Date:",        str(rdate))
    field(W/2+10, iy-14, "Assistant 1:", asst1)
    field(W/2+10, iy-30, "Assistant 2:", asst2)

    # Enhancement
    ey = iy - 68
    cv.setFillColor(HexColor("#1565c0")); cv.setFont("Helvetica-Bold", 9)
    cv.drawString(M, ey, "Enhancement:")
    ex = M + 95
    for label, checked in [("Eye Deviation", eye), ("Photic Stimulation", pho), ("Bioptron", bio)]:
        cv.setStrokeColor(HexColor("#1565c0")); cv.setLineWidth(1)
        cv.rect(ex, ey-2, 9, 9, fill=0, stroke=1)
        if checked:
            cv.setFillColor(HexColor("#1565c0")); cv.rect(ex+1, ey-1, 7, 7, fill=1, stroke=0)
        cv.setFillColor(HexColor("#212121")); cv.setFont("Helvetica", 8.5)
        cv.drawString(ex+13, ey, label)
        ex += 108

    # Divider
    dv = ey - 16
    cv.setStrokeColor(HexColor("#bbdefb")); cv.setLineWidth(0.8)
    cv.line(M, dv, W-M, dv)

    # Images
    iw = (W - 2*M - 20) / 3
    ih = iw * 0.85
    itop = dv - 14

    for i, (key, lbl) in enumerate(zip(["mid","left","right"], ["Mid Frontal","Left Frontal","Right Frontal"])):
        ix = M + i*(iw+10)
        ib = itop - ih
        cv.setFillColor(HexColor("#1565c0")); cv.setFont("Helvetica-Bold", 9)
        cv.drawCentredString(ix+iw/2, itop+2, lbl)
        if key in imgs:
            try:
                cv.drawImage(pil_to_reportlab(imgs[key]), ix, ib, width=iw, height=ih, preserveAspectRatio=False)
                cv.setStrokeColor(HexColor("#90caf9")); cv.setLineWidth(1)
                cv.rect(ix, ib, iw, ih, fill=0, stroke=1)
            except:
                cv.setFillColor(HexColor("#e3f2fd")); cv.rect(ix, ib, iw, ih, fill=1, stroke=0)
        else:
            cv.setFillColor(HexColor("#f5f5f5")); cv.setStrokeColor(HexColor("#bdbdbd"))
            cv.setLineWidth(0.5); cv.rect(ix, ib, iw, ih, fill=1, stroke=1)
            cv.setFillColor(HexColor("#9e9e9e")); cv.setFont("Helvetica", 8)
            cv.drawCentredString(ix+iw/2, ib+ih/2, "No image")
        cv.setStrokeColor(HexColor("#bbdefb")); cv.setLineWidth(0.5)
        cv.line(ix, ib-6, ix+iw, ib-6)

    # Impression
    imp_y = itop - ih - 26
    cv.setFillColor(HexColor("#1565c0")); cv.setFont("Helvetica-Bold", 10)
    cv.drawString(M, imp_y, "Impression:")
    bh = 105
    cv.setFillColor(HexColor("#fafafa")); cv.setStrokeColor(HexColor("#bbdefb")); cv.setLineWidth(0.8)
    cv.roundRect(M, imp_y-bh-8, W-2*M, bh, 5, fill=1, stroke=1)

    if imp:
        cv.setFillColor(HexColor("#212121")); cv.setFont("Helvetica", 9)
        words = imp.split(); line = ""; ly = imp_y - 22
        for word in words:
            test = (line+" "+word).strip()
            if cv.stringWidth(test, "Helvetica", 9) < W-2*M-20:
                line = test
            else:
                cv.drawString(M+8, ly, line); ly -= 13; line = word
                if ly < imp_y - bh: break
        if line: cv.drawString(M+8, ly, line)

    # Footer
    cv.setFillColor(HexColor("#0d47a1")); cv.rect(0, 0, W, 22, fill=1, stroke=0)
    cv.setFillColor(HexColor("#bbdefb")); cv.setFont("Helvetica", 6.5)
    cv.drawString(M, 8, "Haemo-Encephalography Report  ·  Dr. Hany Elhennawy Psychiatric Center")
    cv.drawRightString(W-M, 8, f"Wijdan Therapy  ·  {rdate}")

    cv.save(); buf.seek(0)
    return buf.getvalue()

# ── Status & Button ───────────────────────────────────────────────────────────
st.markdown("---")
missing = []
if not patient_name: missing.append("patient name")
for k in ["mid","left","right"]:
    if k not in processed: missing.append(labels[k] + " photo")
if missing:
    st.info(f"⏳ Still needed: {', '.join(missing)}")

if st.button("🖨️  Generate HEG Report PDF"):
    if missing:
        st.error(f"Please complete: {', '.join(missing)}")
    else:
        with st.spinner("Applying color map and generating report..."):
            pdf_bytes = generate_pdf(
                patient_name, report_date, assistant1, assistant2,
                eye_dev, photic, bioptron, processed, impression
            )
        st.markdown('<div class="success-box">✅ Report generated successfully!</div>', unsafe_allow_html=True)
        st.download_button(
            label="⬇️  Download PDF Report",
            data=pdf_bytes,
            file_name=f"HEG_{patient_name.replace(' ','_')}_{report_date}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

st.markdown("---")
st.caption("🧠 Wijdan Therapy · Dr. Hany Elhennawy Psychiatric Center · HEG Report System")
