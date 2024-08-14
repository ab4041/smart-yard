import os
import cv2
import datetime
import threading
import numpy as np
from tkinter import Tk, Button, Label, Entry, filedialog, Frame, PhotoImage, messagebox
from PIL import Image, ImageTk
from fpdf import FPDF

# Initialize the YOLO model
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Global variables
cap = None
running = False
log_data_list = []

def log_data(truck_id, position, object_size, anomaly=False):
    status = "Anomaly" if anomaly else "Normal"
    log_entry = f"{datetime.datetime.now()}, Truck ID: {truck_id}, Position: {position}, Object Size: {object_size}, Status: {status}\n"
    log_data_list.append(log_entry)
    with open('log_file.txt', 'a') as log_file:
        log_file.write(log_entry)

def detect_objects(frame):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids, confidences, boxes = [], [], []

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

    detected_objects, anomaly_detected = [], False

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(class_ids[i])
            color = (0, 255, 0)  # Default green for normal

            # Simulated rule for detecting anomalies: 
            # Assuming 'person' (class_id 0) is an anomaly for this example
            if label == "0":  # Replace "0" with your anomaly class ID
                color = (0, 0, 255)  # Red for anomaly
                anomaly_detected = True

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {int(confidences[i] * 100)}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            detected_objects.append((label, (x, y, w, h)))

    for obj in detected_objects:
        log_data(truck_id=1, position=obj[1], object_size="medium", anomaly=anomaly_detected)

    return frame

def update_frame(frame_label):
    global cap, running
    if not running:
        return

    ret, frame = cap.read()
    if not ret:
        cap.release()
        return

    frame = detect_objects(frame)
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    frame_label.imgtk = imgtk
    frame_label.config(image=imgtk)

    frame_label.after(10, update_frame, frame_label)

def process_video(video_path=None):
    global cap, running
    if running:
        cap.release()
        running = False

    running = True
    if video_path:
        cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture(0)

    # Create a new window to show video feed
    video_window = Tk()
    video_window.title("Live Video Feed")

    frame_label = Label(video_window)
    frame_label.pack()

    update_frame(frame_label)
    video_window.mainloop()

def stop_video():
    global running
    running = False

def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        frame = cv2.imread(file_path)
        frame = detect_objects(frame)
        processed_image_path = "uploaded_image_processed.jpg"
        cv2.imwrite(processed_image_path, frame)
        messagebox.showinfo("Success", f"Image processed and saved as {processed_image_path}")

def capture_image():
    global cap
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if ret:
        frame = detect_objects(frame)
        processed_image_path = "captured_image.jpg"
        cv2.imwrite(processed_image_path, frame)
        messagebox.showinfo("Success", f"Image captured and processed, saved as {processed_image_path}")

def generate_report():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Truck Monitoring Report", ln=True, align='C')

    with open('log_file.txt') as log_file:
        for line in log_file:
            pdf.cell(200, 10, txt=line, ln=True)

    file_path = filedialog.asksaveasfilename(defaultextension=".pdf",
                                             filetypes=[("PDF files", "*.pdf")],
                                             title="Save PDF Report")

    if file_path:
        pdf.output(file_path)
        messagebox.showinfo("Success", f"PDF Report generated and saved at {file_path}!")

def check_login():
    username = username_entry.get()
    password = password_entry.get()
    if username == "admin" and password == "password":
        login_frame.pack_forget()
        welcome_page()
    else:
        error_label.config(text="Invalid credentials, please try again.")

def welcome_page():
    welcome_frame = Frame(root)
    welcome_frame.pack(fill="both", expand=True)

    label = Label(welcome_frame, text="Welcome to the Walmart Supply Chain Object Detection System", font=("Helvetica", 18))
    label.pack(pady=20)

    description = Label(welcome_frame, text="This system helps in detecting anomalies in trucks within the Walmart supply chain using advanced object detection techniques.", wraplength=400, justify="center")
    description.pack(pady=10)

    start_button = Button(welcome_frame, text="Start Detection", command=lambda: [welcome_frame.pack_forget(), main_interface()])
    start_button.pack(pady=20)

def main_interface():
    main_frame = Frame(root)
    main_frame.pack(fill="both", expand=True)

    label = Label(main_frame, text="Walmart Supply Chain Object Detection System", font=("Helvetica", 18))
    label.pack(pady=10)

    upload_image_button = Button(main_frame, text="Upload Image", command=upload_image)
    upload_image_button.pack(pady=5)

    capture_image_button = Button(main_frame, text="Capture Image", command=capture_image)
    capture_image_button.pack(pady=5)

    capture_webcam_button = Button(main_frame, text="Capture from Webcam", command=lambda: threading.Thread(target=process_video).start())
    capture_webcam_button.pack(pady=5)

    stop_button = Button(main_frame, text="Stop Video Feed", command=stop_video)
    stop_button.pack(pady=5)

    generate_report_button = Button(main_frame, text="Generate Report", command=generate_report)
    generate_report_button.pack(pady=5)

def start_gui():
    global root, username_entry, password_entry, error_label, login_frame
    root = Tk()
    root.title("Walmart Supply Chain Object Detection System")
    root.geometry("500x600")

    # Login Frame
    login_frame = Frame(root)
    login_frame.pack(fill="both", expand=True)

    logo = PhotoImage(file="walmart_logo.png")  # Replace with the correct path to Walmart logo
    logo_label = Label(login_frame, image=logo)
    logo_label.pack(pady=20)

    Label(login_frame, text="Username").pack(pady=5)
    username_entry = Entry(login_frame)
    username_entry.pack(pady=5)

    Label(login_frame, text="Password").pack(pady=5)
    password_entry = Entry(login_frame, show="*")
    password_entry.pack(pady=5)

    error_label = Label(login_frame, text="", fg="red")
    error_label.pack(pady=5)

    Button(login_frame, text="Login", command=check_login).pack(pady=20)

    root.mainloop()

# Start the GUI
if __name__ == "__main__":
    start_gui()
