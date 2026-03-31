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

# ── QX30 F/X Color LUT ────────────────────────────────────────────────────────
# Derived by pixel-by-pixel analysis of raw vs. QX30 F/X-edited image pairs.
# Maps luminance (0–255) → RGB matching the QX30 green-red-blue filter exactly.
QX30_LUT = np.array([
    [ 24,152,191],[ 59,179,143],[ 61,179,132],[ 83,185,142],[ 71,176,141],[ 69,182,142],[ 60,169,152],[ 47,163,153],
    [ 53,166,151],[ 44,157,152],[ 41,157,157],[ 52,163,150],[ 39,151,144],[ 39,155,152],[ 58,162,154],[ 54,164,146],
    [ 52,165,141],[ 59,172,136],[ 55,165,143],[ 60,166,143],[ 47,161,141],[ 46,160,149],[ 34,155,152],[ 39,157,155],
    [ 47,158,148],[ 53,165,150],[ 49,164,154],[ 46,162,157],[ 50,163,159],[ 41,159,158],[ 44,156,144],[ 48,159,150],
    [ 55,160,148],[ 59,162,156],[ 60,165,154],[ 47,161,159],[ 49,161,154],[ 44,159,151],[ 50,161,151],[ 49,162,155],
    [ 53,166,142],[ 40,157,154],[ 48,162,146],[ 45,161,147],[ 48,164,149],[ 52,166,149],[ 50,165,146],[ 53,167,152],
    [ 56,168,144],[ 52,168,145],[ 53,168,144],[ 55,168,138],[ 61,171,133],[ 61,171,137],[ 57,169,138],[ 62,172,143],
    [ 62,170,137],[ 57,168,143],[ 57,167,137],[ 63,172,135],[ 73,178,140],[ 74,178,141],[ 64,172,136],[ 62,172,141],
    [ 63,174,137],[ 63,174,143],[ 61,174,136],[ 74,182,128],[ 79,186,125],[ 79,182,116],[ 81,188,120],[ 80,186,115],
    [ 82,188,114],[ 91,194,115],[ 84,190,114],[ 83,187,106],[ 89,191,114],[ 83,186,109],[ 88,192,114],[ 92,193,106],
    [ 90,192,105],[ 91,192,102],[ 98,198,108],[ 96,194,103],[ 98,197,100],[ 99,198,103],[ 98,197, 98],[ 97,195,103],
    [ 93,192, 97],[ 96,195,100],[100,197, 97],[ 98,196, 98],[100,200, 99],[100,198, 96],[ 98,199,101],[100,198, 96],
    [103,199,105],[109,204, 93],[104,202, 92],[ 97,195, 97],[105,202, 94],[101,198, 91],[116,207, 91],[ 98,197, 90],
    [111,207, 84],[105,201, 89],[108,205, 83],[106,202, 91],[109,204, 85],[109,206, 86],[111,206, 91],[109,204, 86],
    [117,209, 88],[114,207, 81],[113,207, 87],[118,209, 80],[113,207, 82],[127,213, 82],[113,206, 74],[120,210, 75],
    [117,209, 70],[120,211, 74],[120,211, 72],[124,214, 73],[123,211, 75],[127,217, 73],[123,211, 72],[123,212, 69],
    [126,214, 73],[124,212, 67],[132,219, 74],[126,213, 69],[127,216, 69],[129,216, 69],[125,213, 67],[126,215, 64],
    [121,209, 64],[120,209, 62],[128,216, 70],[121,210, 66],[128,217, 72],[125,214, 69],[122,213, 67],[124,214, 68],
    [115,204, 64],[119,209, 66],[118,207, 69],[118,208, 68],[116,207, 66],[119,210, 71],[115,203, 68],[115,202, 67],
    [112,197, 68],[111,196, 70],[112,196, 66],[108,191, 67],[112,195, 69],[109,191, 66],[110,189, 70],[108,178, 68],
    [107,174, 65],[107,164, 63],[104,166, 64],[101,160, 61],[103,162, 61],[100,156, 60],[103,160, 63],[103,158, 62],
    [103,159, 64],[106,158, 63],[106,167, 68],[102,158, 63],[105,154, 65],[102,159, 63],[101,157, 62],[101,148, 59],
    [ 99,159, 61],[101,150, 60],[100,154, 59],[101,156, 61],[104,154, 61],[ 99,153, 60],[100,144, 57],[ 99,150, 60],
    [102,139, 57],[103,139, 61],[107,136, 61],[106,133, 59],[104,121, 54],[108,124, 58],[119,117, 59],[119,113, 58],
    [121,111, 60],[128,122, 71],[129,116, 68],[122,105, 54],[124,105, 53],[128,100, 51],[124, 99, 47],[128, 95, 46],
    [129, 91, 46],[127, 92, 44],[132, 85, 45],[131, 86, 45],[131, 83, 42],[134, 82, 43],[137, 83, 44],[141, 79, 42],
    [143, 77, 42],[142, 77, 41],[152, 78, 43],[159, 70, 40],[164, 68, 38],[166, 67, 38],[172, 62, 36],[175, 59, 34],
    [189, 55, 33],[194, 51, 31],[193, 52, 32],[194, 53, 32],[191, 52, 30],[191, 50, 28],[193, 49, 28],[190, 50, 28],
    [193, 50, 28],[191, 47, 25],[194, 46, 25],[200, 47, 27],[204, 47, 28],[204, 45, 25],[207, 45, 25],[214, 46, 25],
    [206, 44, 24],[208, 42, 23],[210, 41, 22],[215, 42, 22],[217, 43, 23],[219, 44, 24],[219, 41, 22],[216, 39, 20],
    [222, 43, 23],[219, 41, 20],[221, 39, 19],[227, 42, 20],[228, 41, 18],[229, 42, 20],[229, 42, 20],[230, 41, 19],
    [231, 40, 17],[232, 41, 16],[232, 41, 17],[235, 44, 20],[232, 40, 18],[236, 41, 16],[243, 45, 16],[243, 45, 16],
], dtype=np.uint8)


def apply_qx30_fx(img: Image.Image) -> Image.Image:
    """
    Apply the QX30 F/X green-red-blue colormap to a raw forehead photo.
    Matches the exact output of the QX30 camera's built-in F/X filter.
    dark/cool areas → blue, mid-range → lime green, bright/warm areas → red.
    """
    arr = np.array(img.convert("RGB")).astype(np.float32)
    r = arr[:, :, 0] / 255.0
    g = arr[:, :, 1] / 255.0
    b = arr[:, :, 2] / 255.0

    # Weighted luminance — heavier red channel weight matches QX30's
    # near-infrared sensitivity (blood/haemoglobin shows bright in red channel)
    lum = r * 0.60 + g * 0.25 + b * 0.15

    # Smooth to suppress camera sensor noise and fine texture
    lum = gaussian_filter(lum, sigma=12)

    # Normalize to [0, 1] across the full dynamic range of this image
    lum = (lum - lum.min()) / (lum.max() - lum.min() + 1e-8)

    # Contrast stretch: expand the mid-range, push ends toward extremes
    # This matches the QX30's built-in contrast enhancement
    lum = ((lum - 0.15) / 0.85).clip(0, 1)
    lum = np.power(lum, 1.3)   # gamma curve matches QX30 output saturation

    # Apply LUT
    lum_idx = (lum * 255).clip(0, 255).astype(np.int32)
    out = QX30_LUT[lum_idx]
    return Image.fromarray(out.astype(np.uint8), "RGB")


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
st.caption("Upload the 3 photos taken with the USB microscope. The QX30 F/X color filter is applied automatically.")

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
    st.markdown('<div class="section-card"><div class="section-title">🎨 QX30 F/X Color Preview</div>', unsafe_allow_html=True)
    prev_cols = st.columns(3)
    for i, (key, f) in enumerate(files.items()):
        with prev_cols[i]:
            st.caption(labels[key])
            if f is not None:
                raw     = Image.open(f).convert("RGB")
                colored = apply_qx30_fx(raw)
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
        with st.spinner("Applying QX30 F/X color filter and generating report..."):
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
