import streamlit as st
import cv2
from ultralytics import YOLO
import easyocr
import numpy as np
from PIL import Image

model = YOLO("best.pt") 

reader = easyocr.Reader(['en'])

st.markdown("<h2 style='text-align: center;'>🚘 Automatic License Plate Recognition</h2>", unsafe_allow_html=True)
st.markdown("<h6 style='text-align: center;'>Upload an image of a car to detect and extract the license plate text.</h6>", unsafe_allow_html=True)


uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    img = np.array(image)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Detect with YOLO
    results = model.predict(img, conf=0.5)

    for r in results:
        for box in r.boxes.xyxy: 
            x1, y1, x2, y2 = map(int, box.tolist())
            plate = img[y1:y2, x1:x2]

   
            st.image(plate, caption="🔹 Detected Plate", use_column_width=False)

    
            gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)  
            blur = cv2.GaussianBlur(gray, (3,3), 0)        
            thresh = cv2.adaptiveThreshold(
                blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )  
            resized = cv2.resize(
                thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC
            )  

            
            st.image(resized, caption="✨ Enhanced Plate (Preprocessed)", use_column_width=False)

          
            text = reader.readtext(resized)
            extracted_text = [t[1] for t in text]

            st.subheader("📖 Extracted Plate Text:")
            if extracted_text:
                st.success(" ".join(extracted_text))
            else:
                st.warning("⚠️ No text detected.")
