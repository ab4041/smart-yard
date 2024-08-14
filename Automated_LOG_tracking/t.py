import os
import cv2
import datetime
from tkinter import Tk, Button, Label
from fpdf import FPDF
import threading
import numpy as np
import matplotlib.pyplot as plt

# Initialize the YOLO model
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

log_data_list = []
cap = None
running = True  # Global variable to control the video feed

# Dictionary to map class IDs to their respective colors and labels
class_map = {
    0: ("Person", (0, 0, 255)),  # Red for person
    1: ("Light", (0, 165, 255)),  # Orange for light
    2: ("Truck", (0, 255, 0))    # Green for trucks
}

def log_data(truck_id, position, object_size, label, anomaly=False):
    status = "Anomaly" if anomaly else "Normal"
    log_entry = f"{datetime.datetime.now()}, Truck ID: {truck_id}, Position: {position}, Object Size: {object_size}, Label: {label}, Status: {status}\n"
    log_data_list.append(log_entry)
    with open('log_file.txt', 'a') as log_file:
        log_file.write(log_entry)


def detect_objects(frame):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

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

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    detected_objects = []
    anomaly_detected = False

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label, color = class_map.get(class_ids[i], ("Unknown", (255, 255, 255)))

            # Determine if it's an anomaly
            if label == "Person" or label == "Light":
                anomaly_detected = True

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {int(confidences[i] * 100)}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            detected_objects.append((label, (x, y, w, h)))

    for obj in detected_objects:
        log_data(truck_id=1, position=obj[1], object_size="medium", label=obj[0], anomaly=anomaly_detected)

    return frame


def capture_image():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if ret:
        frame = detect_objects(frame)
        cv2.imwrite("captured_image.jpg", frame)
        print("Image captured and processed.")


def display_frame(frame):
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.pause(0.001)  # Pause for a short while to display the image


def process_video(video_path=None):
    global cap, running
    if video_path:
        cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture(0)

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        frame = detect_objects(frame)
        display_frame(frame)  # Use matplotlib to display the frame

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    plt.close()  # Close the plot window when done


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

    pdf_output_path = os.path.join(os.getcwd(), "report.pdf")
    pdf.output(pdf_output_path)
    print(f"PDF Report Generated and saved at {pdf_output_path}!")


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
