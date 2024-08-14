import os
import cv2
import datetime
from tkinter import Tk, Button, Label, messagebox, PhotoImage
from tkinter import Canvas
from fpdf import FPDF
import threading
import numpy as np

# Initialize the YOLO model
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Define the classes YOLO can detect
classes = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
    "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
    "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "TV", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]

log_data_list = []
cap = None
running = True  # Global variable to control the video feed

def log_data(truck_id, position, object_size, label):
    log_entry = f"{datetime.datetime.now()}, Truck ID: {truck_id}, Position: {position}, Object Size: {object_size}, Detected Object: {label}\n"
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

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]] if class_ids[i] < len(classes) else "Unknown"

            # Assign colors based on the label
            color = (255, 255, 255)  # Default white for unknown objects
            if label == "person":
                color = (0, 0, 255)  # Red for person
            elif label == "traffic light":
                color = (0, 165, 255)  # Orange for traffic light
            elif label == "truck":
                color = (0, 255, 0)  # Green for truck

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {int(confidences[i] * 100)}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            detected_objects.append((label, (x, y, w, h)))

    for obj in detected_objects:
        log_data(truck_id=1, position=obj[1], object_size="medium", label=obj[0])

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
    cv2.imshow('Video Feed', frame)
    cv2.waitKey(1)  # Display frame for 1 ms

def process_video(video_path=None):
    global cap, running
    if video_path:
        cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture(0)

    frame_skip = 2  # Process every 2nd frame
    frame_count = 0

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip == 0:
            frame = detect_objects(frame)
            display_frame(frame)  # Use OpenCV to display the frame

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()  # Close the OpenCV windows when done

def stop_video():
    global running
    running = False

def generate_report():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Truck Monitoring Report", ln=True, align='C')

    try:
        with open('log_file.txt') as log_file:
            for line in log_file:
                # Split the text into lines that fit within the page width
                lines = pdf.multi_cell(0, 10, line)
                pdf.ln()

        pdf_output_path = os.path.join(os.getcwd(), "report.pdf")
        pdf.output(pdf_output_path)
        messagebox.showinfo("Success", f"PDF Report Generated and saved at {pdf_output_path}!")
    except PermissionError:
        messagebox.showerror("File Error", "Permission denied: 'D:\\Waltmart\\report.pdf'. Please close the file if it is open or check permissions.")

def start_gui():
    root = Tk()
    root.title("Object Detection and Reporting")

    # Set Walmart background image
    bg_image = PhotoImage(file="walmart_bg.png")
    canvas = Canvas(root, width=bg_image.width(), height=bg_image.height())
    canvas.pack(fill="both", expand=True)
    canvas.create_image(0, 0, image=bg_image, anchor="nw")

    # Add widgets
    label = Label(root, text="Choose an Option:", font=("Arial", 16), bg="white", fg="blue")
    canvas.create_window(300, 50, anchor="nw", window=label)

    capture_image_button = Button(root, text="Capture Image", command=capture_image)
    canvas.create_window(300, 100, anchor="nw", window=capture_image_button)

    capture_webcam_button = Button(root, text="Capture from Webcam", command=lambda: threading.Thread(target=process_video).start())
    canvas.create_window(300, 150, anchor="nw", window=capture_webcam_button)

    stop_button = Button(root, text="Stop Video Feed", command=stop_video)
    canvas.create_window(300, 200, anchor="nw", window=stop_button)

    generate_report_button = Button(root, text="Generate Report", command=generate_report)
    canvas.create_window(300, 250, anchor="nw", window=generate_report_button)

    root.mainloop()

if __name__ == '__main__':
    start_gui()
