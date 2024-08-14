import streamlit as st
import cv2
import os
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import json
from ultralytics import YOLO
# import easyocr

# Set the environment variable to avoid OpenMP runtime conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from paddleocr import PaddleOCR
ocr = PaddleOCR(lang='en',rec_algorithm='CRNN')


# Load the YOLOv8 model
model = YOLO('best.pt')  # Replace with your specific YOLOv8 model path if needed
# Initialize EasyOCR reader
# reader = easyocr.Reader(['en'])

def enhance_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    enhanced_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_image

def adaptive_threshold(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
    return binary_image

def load_image(image_file):
    img = Image.open(image_file)
    return img

def predict_license_plate(image):
    # Convert PIL image to numpy array
    image_np = np.array(image)
    # Perform YOLOv8 detection
    results = model.predict(image_np)

    annotations = []
    for result in results:
        for detection in result.boxes:
            bbox = detection.xyxy[0].tolist()  # Bounding box coordinates
            confidence = detection.conf[0].item()  # Confidence score
            class_id = detection.cls[0].item()  # Class ID
            
            # For this example, we assume class_id = 0 corresponds to license plates
            if class_id == 0:
                annotations.append({
                    'bbox': bbox,
                    'confidence': confidence
                })
    
    return annotations

def draw_annotations(image, annotations):
    draw = ImageDraw.Draw(image)
    for ann in annotations:
        bbox = ann['bbox']
        confidence = ann['confidence']
        
        # Draw bounding box
        draw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])], outline='blue', width=3)
        # Annotate confidence
        # draw.text((bbox[0], bbox[1] - 10), f"Confidence: {confidence:.2f}", fill='blue')
        font=ImageFont.load_default()

        cropped_plate = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))

        # Convert cropped Pillow image back to NumPy array
        cropped_plate_arr = np.array(cropped_plate)

        # Preprocess the image
        enhanced_image = enhance_contrast(cropped_plate_arr)
        binary_image = adaptive_threshold(enhanced_image)

        # results = reader.readtext(binary_image)
        results = ocr.ocr(binary_image, cls=False, det=False)
        
        if results and results[0][0][1]>0.6:
            label=results[0][0][0]

        else:
            label=f"License_plate {confidence}"
        
        ann['ocr_text'] = label

        w,h=font.getbbox(label)[2:]
        draw.rectangle((bbox[0], bbox[1]-h, bbox[0]+w, bbox[1]), fill='blue')
        draw.text((bbox[0],bbox[1]-h),label,fill='white',font=font)
        
    
    return image

st.title("License Plate Recognition")
st.write("Upload an image to detect and annotate license plates.")

# Image upload
image_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if image_file is not None:
    # Load and display the uploaded image
    image = load_image(image_file)
    
    # Predict license plates
    annotations = predict_license_plate(image)
    
    
    # Annotate image
    annotated_image = draw_annotations(image.copy(), annotations)


    # Display JSON output
    st.subheader("Detection Results (JSON)")
    st.json(annotations)
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Uploaded Image")
        st.image(image, caption='Uploaded Image', use_column_width=True)
    
    with col2:
        st.subheader("Annotated Image")
        st.image(annotated_image, caption='Annotated Image', use_column_width=True)
