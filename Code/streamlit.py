import os
import tensorflow as tf
import streamlit as st
import time
import tempfile
import cv2
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings
# Suppress TensorFlow logging (2)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

st.title("Tensorflow Object Detection App")
st.write("Upload an image or a video and see the object detection results using the TensorFlow model.")
option = st.selectbox("Select input type:", ("Image", "Video", "Webcam"))
st.write("You selected:", option)
uploaded_file = None
if option != "Webcam":
    uploaded_file = st.file_uploader("Upload image or video", type=["jpg", "png", "mp4"])

# PROVIDE PATH TO MODEL DIRECTORY
PATH_TO_MODEL_DIR = '../Code/content/exported_models/ssd_20ks_model'

# PROVIDE PATH TO LABEL MAP
PATH_TO_LABELS = '../models/Top-Damage_label_map.pbtxt'

# PROVIDE THE MINIMUM CONFIDENCE THRESHOLD
MIN_CONF_THRESH = float(0.10)  # Adjust this value according to your needs


#@st.cache_resource cache the function and load_model() and load model 
@st.cache_resource
def load_model():
    print('Loading model...', end='')
    start_time = time.time()

    # LOAD SAVED MODEL AND BUILD DETECTION FUNCTION
    detect_fn = tf.saved_model.load(PATH_TO_MODEL_DIR + "/saved_model")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done! Took {} seconds'.format(elapsed_time))
    
    return detect_fn

detect_fn = load_model()

# LOAD LABEL MAP DATA FOR PLOTTING
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.
    Puts image into numpy array to feed into TensorFlow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.
    Args:
      path: the file path to the image
    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))

def process_video(video_bytes):
    with tempfile.NamedTemporaryFile(suffix=".mp4") as temp_file:
        temp_file.write(video_bytes)
        temp_file.seek(0)
        cap = cv2.VideoCapture(temp_file.name)
        frame_placeholder = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            intime= time.time()
            # Process frame using object detection model
            detections = detect_objects(frame)
            outtime = time.time()
            print("Inference:",(outtime-intime)*1000,"ms")

            # Visualize detections on frame
            frame_with_detections = visualize_detections(frame, detections)

            # Display the resulting frame
            frame_placeholder.image(frame_with_detections, caption="Predicted frame with bounding boxes",
                                     use_column_width=True)

        # Release resources
        cap.release()

def detect_objects(frame):
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(frame)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    return detections

def visualize_detections(frame, detections):
    frame_with_detections = frame.copy()

    # Filter out bounding boxes with confidence scores below MIN_CONF_THRESH
    filtered_boxes = detections['detection_boxes'][detections['detection_scores'] >= MIN_CONF_THRESH]
    filtered_classes = detections['detection_classes'][detections['detection_scores'] >= MIN_CONF_THRESH]
    filtered_scores = detections['detection_scores'][detections['detection_scores'] >= MIN_CONF_THRESH]

    # Apply Non-Maximum Suppression
    selected_indices = tf.image.non_max_suppression(filtered_boxes, filtered_scores, max_output_size=50, iou_threshold=0.5)
    selected_boxes = tf.gather(filtered_boxes, selected_indices)
    selected_classes = tf.gather(filtered_classes, selected_indices)
    selected_scores = tf.gather(filtered_scores, selected_indices)

    # Visualization of the results of a detection.
    for i in range(selected_boxes.shape[0]):
        class_id = int(selected_classes[i])
        score = selected_scores[i].numpy()
        label = '{}: {:.2f}'.format(category_index[class_id]['name'], score)

        # Convert box to pixel coordinates
        height, width, _ = frame_with_detections.shape
        box = selected_boxes[i].numpy()
        box_pixels = [int(box[0] * height), int(box[1] * width), int(box[2] * height), int(box[3] * width)]
        start_point = (box_pixels[1], box_pixels[0])  # (xmin, ymin)
        end_point = (box_pixels[3], box_pixels[2])    # (xmax, ymax)

        # Set color based on class_list
        if category_index[class_id]['name'] == "flangeWithGasket":
            color = (0, 255, 0)  # Green
        else:
            color = (255, 0, 0)  # Red

        # Draw the bounding box
        thickness = 5  # Increase for thicker box
        cv2.rectangle(frame_with_detections, start_point, end_point, color, thickness)

        # Draw label and score
        font_scale = 2  # Adjust for font size
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_color = (0, 0, 255)
        cv2.putText(frame_with_detections, label, (box_pixels[1], box_pixels[0] - 20), font, font_scale, text_color, thickness)

    return frame_with_detections

if option == "Webcam":  # Webcam selected
    video_input = 0  # Webcam source
    cap = cv2.VideoCapture(video_input)
    frame_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame using object detection model
        detections = detect_objects(frame)

        # Visualize detections on frame
        frame_with_detections = visualize_detections(frame, detections)

        # Display the resulting frame
        frame_placeholder.image(frame_with_detections, caption="Predicted frame with bounding boxes",
                                 use_column_width=True)

    # Release resources
    cap.release()
elif option == "Image":
    if uploaded_file is not None:
        image = load_image_into_numpy_array(uploaded_file)
        # Process image using object detection model
        detections = detect_objects(image)
        # Visualize detections on image
        image_with_detections = visualize_detections(image, detections)
        # Display the resulting image
        st.image(image_with_detections, caption="Predicted image with bounding boxes", use_column_width=True)
elif option == "Video":
    if uploaded_file is not None:
        video_bytes = uploaded_file.read()
        process_video(video_bytes)
else:
    st.write("Please upload a file or select Webcam.")
