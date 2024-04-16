import os
import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from io import BytesIO
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("../../models/best_v8_8th_April.pt")

def infer_on_image(image_path):
    # Perform prediction on an image
    results = model.predict(image_path, conf=0.1)
    result = results[0]

    # Convert the image from numpy array to a PIL Image
    pil_image = Image.fromarray(result.plot()[:, :, ::-1])

    # Resize the image
    desired_width = 600  # Set your desired width
    desired_height = 600  # Set your desired height
    pil_image = pil_image.resize((desired_width, desired_height), Image.LANCZOS)

    # Convert PIL Image to BytesIO
    image_bytes = BytesIO()
    pil_image.save(image_bytes, format='PNG')
    image_bytes.seek(0)

    # Display the image using Tkinter
    root = tk.Tk()
    root.title("YOLO Prediction Result")

    # Convert BytesIO to Tkinter PhotoImage
    image_tk = ImageTk.PhotoImage(Image.open(image_bytes))

    # Create a label with the image
    label = tk.Label(root, image=image_tk)
    label.pack()

    # Set window size
    window_width = desired_width + 10
    window_height = desired_height + 10
    root.geometry(f"{window_width}x{window_height}")

    root.mainloop()

def infer_on_video(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame to 600x600
        frame = cv2.resize(frame, (600, 600))

        # Perform prediction on the frame
        results = model.predict(frame, conf=0.1)
        result = results[0]

        # Convert the image from numpy array to a PIL Image
        pil_image = Image.fromarray(result.plot()[:, :, ::-1])

        # Convert PIL Image to OpenCV format
        frame_with_overlay = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # Display the image using OpenCV
        cv2.imshow("YOLO Prediction Result", frame_with_overlay)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close the window when done
    cap.release()
    cv2.destroyAllWindows()

def infer_on_input(input_path):
    if os.path.isfile(input_path):
        if input_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            infer_on_image(input_path)
        elif input_path.lower().endswith(('.mp4', '.avi', '.mov')):
            infer_on_video(input_path)
        else:
            print("Unsupported file format.")
    else:
        print("Input file does not exist.")

if __name__ == "__main__":
    input_path = input("Enter the path to image or video: ")
    infer_on_input(input_path)
