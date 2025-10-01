# 🚘 Automatic License Plate Recognition 

This project is an **Automatic License Plate Recognition ** system built using **YOLOv8** for object detection and **EasyOCR** for text recognition.  
It allows users to upload a car image, automatically detects the license plate, applies preprocessing techniques, and extracts the license plate number.

🔗 **Live Demo:** [Streamlit App](https://car-license-plate-detection.streamlit.app/)

---

## 📌 Features
- 🖼️ Upload a car image directly from your device.  
- 🔍 Detects and crops the license plate region using **YOLOv8**.  
- 🧹 Enhances the plate image (grayscale, Gaussian blur, adaptive thresholding, resizing).  
- 🔠 Extracts the text from the license plate using **EasyOCR**.  
- ✅ Displays the **largest/most relevant detected text** as the license plate number.  

---

## 🛠️ Tech Stack
- **Python 3.11+**
- [YOLOv8 (Ultralytics)](https://github.com/ultralytics/ultralytics)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [OpenCV](https://opencv.org/)
- [Streamlit](https://streamlit.io/)
- **NumPy, Pillow**

---

## 📂 Project Structure
```
├── app.py               # Streamlit app (main file)
├── best.pt              # Trained YOLOv8 weights (license plate detector)
├── requirements.txt     # Required dependencies
└── README.md            # Project documentation
```

---

## 🚀 Installation & Usage

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/car-license-plate-detection.git
   cd car-license-plate-detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

4. **Upload an image** and get the detected license plate text.

---

## 📋 Requirements
Example `requirements.txt`:
```
streamlit
ultralytics
easyocr
opencv-python-headless
pillow
numpy
```

---

## 📸 Demo
- Upload a car image  
- The system detects the plate  
- Preprocesses and extracts the text  

---

## 📌 Future Improvements
- Add support for **Arabic license plates**.  
- Improve OCR accuracy with **Tesseract / PaddleOCR** integration.  
- Deploy on other platforms (Heroku, AWS, etc.).  

---

## 👨‍💻 Author
Developed by **[Muhammad Al-Aasar]** 🚀  
If you like this project, don’t forget to ⭐ the repo!
