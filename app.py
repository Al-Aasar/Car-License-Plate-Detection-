import streamlit as st
import cv2
from ultralytics import YOLO
import easyocr
import numpy as np
from PIL import Image

# تحميل الموديل
model = YOLO("best.pt")

# تحميل EasyOCR
reader = easyocr.Reader(['en'])

# عنوان في المنتصف
st.markdown("<h2 style='text-align: center;'>🚘 Automatic License Plate Recognition</h2>", unsafe_allow_html=True)
st.markdown("<h6 style='text-align: center;'>Upload an image of a car to detect and extract the license plate text.</h6>", unsafe_allow_html=True)

# رفع صورة
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # قراءة الصورة
    image = Image.open(uploaded_file)
    img = np.array(image)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    # YOLO detection
    results = model.predict(img, conf=0.5)

    for r in results:
        for box in r.boxes.xyxy: 
            x1, y1, x2, y2 = map(int, box.tolist())
            plate = img[y1:y2, x1:x2]

            # عرض اللوحة المقطوعة
            st.image(plate, caption="🔹 Detected Plate", use_column_width=False)

            # -------- تحسين الصورة قبل OCR --------
            gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)  # رمادي
            blur = cv2.GaussianBlur(gray, (3,3), 0)        # تقليل ضوضاء
            thresh = cv2.adaptiveThreshold(
                blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )  # زيادة التباين
            resized = cv2.resize(
                thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC
            )  # تكبير

            # عرض اللوحة بعد التحسين
            st.image(resized, caption="✨ Enhanced Plate (Preprocessed)", use_column_width=False)

            # OCR
            text = reader.readtext(resized)
            extracted_text = [t[1] for t in text]

            # عرض النص
            st.subheader("📖 Extracted Plate Text:")
            if extracted_text:
                st.success(" ".join(extracted_text))
            else:
                st.warning("⚠️ No text detected.")
