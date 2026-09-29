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

st.markdown("<h2 style='text-align: center;'>🚘 Automatic License Plate Recognition</h2>", unsafe_allow_html=True)
st.markdown("<h6 style='text-align: center;'>Upload an image of a car to detect and extract the license plate text.</h6>", unsafe_allow_html=True)

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

            gray = cv2.cvtColor(plate, cv2.COLOR_RGB2GRAY)
            blur = cv2.GaussianBlur(gray, (3, 3), 0)
            thresh = cv2.adaptiveThreshold(
                blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            resized = cv2.resize(
                thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC
            )

            st.image(resized, caption="✨ Enhanced Plate (Preprocessed)", width="content")

            results_ocr = reader.readtext(resized)

            st.subheader("📖 Extracted Plate Text:")
            if results_ocr:
                results_ocr.sort(
                    key=lambda x: (x[0][2][0] - x[0][0][0]) * (x[0][2][1] - x[0][0][1]),
                    reverse=True
                )
                biggest_text = results_ocr[0][1]
                st.success(biggest_text)
            else:
                st.warning("⚠️ No text detected.")

    if not found:
        st.warning("⚠️ No license plate detected in the image.")
