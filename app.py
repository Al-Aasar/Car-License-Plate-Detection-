import streamlit as st
import cv2
from ultralytics import YOLO
import easyocr
import numpy as np
from PIL import Image


@st.cache_resource
def load_model():
    return YOLO("best.pt")


@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'], gpu=False)


model = load_model()
reader = load_reader()

ALLOWED = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

st.markdown("<h2 style='text-align: center;'>🚘 Automatic License Plate Recognition</h2>", unsafe_allow_html=True)
st.markdown("<h6 style='text-align: center;'>Upload an image of a car to detect and extract the license plate text.</h6>", unsafe_allow_html=True)

crop_margins = st.checkbox("Crop plate margins (European-style plates)", value=True)

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)

    st.image(img, caption="Uploaded Image", width="stretch")

    results = model.predict(img, conf=0.5)

    found = False
    for r in results:
        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box.tolist())
            x1, y1 = max(x1, 0), max(y1, 0)
            plate = img[y1:y2, x1:x2]

            if plate.size == 0:
                continue
            found = True

            st.image(plate, caption="🔹 Detected Plate", width="content")

            # Preprocessing: grayscale + optional margin crop + upscale
            gray = cv2.cvtColor(plate, cv2.COLOR_RGB2GRAY)

            if crop_margins:
                h, w = gray.shape
                gray = gray[int(h * 0.05):int(h * 0.75), int(w * 0.15):int(w * 0.95)]

            resized = cv2.resize(
                gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC
            )

            st.image(resized, caption="✨ Enhanced Plate", width="content")

            # OCR
            results_ocr = reader.readtext(
                resized,
                allowlist=ALLOWED,
                text_threshold=0.6,
                low_text=0.3,
                mag_ratio=1.5,
                width_ths=0.3,
            )

            st.subheader("📖 Extracted Plate Text:")

            # Drop small boxes (e.g., the sticker) relative to the tallest box
            if results_ocr:
                heights = [abs(t[0][2][1] - t[0][0][1]) for t in results_ocr]
                max_h = max(heights)
                results_ocr = [t for t, hh in zip(results_ocr, heights) if hh > 0.6 * max_h]

            if results_ocr:
                # Sort left-to-right, keep confident detections, and join
                results_ocr.sort(key=lambda x: x[0][0][0])
                text = " ".join(t[1] for t in results_ocr if t[2] > 0.3)

                if text:
                    st.success(text)
                else:
                    st.warning("⚠️ Text detected but confidence is too low.")
            else:
                st.warning("⚠️ No text detected.")

    if not found:
        st.warning("⚠️ No license plate detected in the image.")
