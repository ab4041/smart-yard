import os
import cv2
import datetime
from tkinter import Tk, Button, Label, filedialog
from fpdf import FPDF
import threading
import numpy as np
from PIL import Image, ImageTk
import math

# Initialize the YOLO model
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

log_data_list = []
cap = None
running = True  # Global variable to control the video feed
previous_positions = {}

def log_data(truck_id, position, object_size, anomaly=False, turn_info=""):
    status = "Anomaly" if anomaly else "Normal"
    log_entry = f"{datetime.datetime.now()}, Truck ID: {truck_id}, Position: {position}, Object Size: {object_size}, Status: {status}, Turn Info: {turn_info}\n"
    log_data_list.append(log_entry)
    with open('log_file.txt', 'a') as log_file:
        log_file.write(log_entry)

def calculate_turn_angle(prev_position, current_position):
    if prev_position and current_position:
        x1, y1 = prev_position
        x2, y2 = current_position

        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        return angle
    return None

def detect_objects(frame):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    centers = {}

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                centers[class_id] = (center_x, center_y)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    detected_objects = []
    anomaly_detected = False
    turn_info = ""

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(class_ids[i])
            color = (0, 255, 0)  # Default green for normal

            if label == "0":  # Replace "0" with your anomaly class ID
                color = (0, 0, 255)  # Red for anomaly
                anomaly_detected = True

            prev_position = previous_positions.get(label)
            current_position = centers.get(class_ids[i])

            angle = calculate_turn_angle(prev_position, current_position)
            previous_positions[label] = current_position

            if angle is not None:
                if angle > 10:
                    turn_info = f"Turning Right: {angle:.2f}°"
                elif angle < -10:
                    turn_info = f"Turning Left: {angle:.2f}°"
                else:
                    turn_info = "Moving Straight"
                cv2.putText(frame, turn_info, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {int(confidences[i] * 100)}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            detected_objects.append((label, (x, y, w, h)))

    for obj in detected_objects:
        log_data(truck_id=1, position=obj[1], object_size="medium", anomaly=anomaly_detected, turn_info=turn_info)

    return frame

def capture_image():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if ret:
        frame = detect_objects(frame)
        cv2.imwrite("captured_image.jpg", frame)
        print("Image captured and processed.")

def update_frame(frame_label):
    global cap, running
    if not running:
        return

    ret, frame = cap.read()
    if not ret:
        return

    frame = detect_objects(frame)

    # Convert frame to ImageTk format
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    frame_label.imgtk = imgtk

    # Schedule the update_frame function to run again
    frame_label.after(10, update_frame, frame_label)

def process_video(video_path=None):
    global cap, running
    if video_path:
        cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture(0)

    # Create a new window to show video feed
    video_window = Tk()
    video_window.title("Live Video Feed")

    frame_label = Label(video_window)
    frame_label.pack()

    # Start updating the frames
    update_frame(frame_label)
    video_window.mainloop()

def stop_video():
    global running
    running = False

def generate_report():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Truck Monitoring Report", ln=True, align='C')

    with open('log_file.txt') as log_file:
        for line in log_file:
            pdf.cell(200, 10, txt=line, ln=True)

    # Open a save dialog to allow the user to choose where to save the PDF
    file_path = filedialog.asksaveasfilename(defaultextension=".pdf",
                                             filetypes=[("PDF files", "*.pdf")],
                                             title="Save PDF Report")

    if file_path:
        pdf.output(file_path)
        print(f"PDF Report Generated and saved at {file_path}!")

def start_gui():
    root = Tk()
    root.title("Object Detection and Reporting")

    label = Label(root, text="Choose an Option:")
    label.pack(pady=10)

    capture_image_button = Button(root, text="Capture Image", command=capture_image)
    capture_image_button.pack(pady=5)

    capture_webcam_button = Button(root, text="Capture from Webcam", command=lambda: threading.Thread(target=process_video).start())
    capture_webcam_button.pack(pady=5)

    stop_button = Button(root, text="Stop Video Feed", command=stop_video)
    stop_button.pack(pady=5)

    generate_report_button = Button(root, text="Generate Report", command=generate_report)
    generate_report_button.pack(pady=5)

    root.mainloop()

if __name__ == '__main__':
    start_gui()
