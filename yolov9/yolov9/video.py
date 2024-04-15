import os
import tensorflow as tf
import time
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

# PROVIDE PATH TO MODEL DIRECTORY
# PROVIDE PATH TO MODEL DIRECTORY
PATH_TO_MODEL_DIR = './content/exported_models/ssd_20ks_model'

# PROVIDE PATH TO LABEL MAP
PATH_TO_LABELS = '../../models/Top-Damage_label_map.pbtxt'

VIDEO_PATH = "../output/testVideo/test.mp4"

# PROVIDE THE MINIMUM CONFIDENCE THRESHOLD
MIN_CONF_THRESH = float(0.10)

# LOAD THE MODEL
PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

print('Loading model...', end='')
start_time = time.time()

# LOAD SAVED MODEL AND BUILD DETECTION FUNCTION
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

# LOAD LABEL MAP DATA FOR PLOTTING
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)

# Function to process video
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame using object detection model
        detections = detect_objects(frame)

        # Visualize detections on frame
        frame_with_detections = visualize_detections(frame, detections)

        # Display the resulting frame
        cv2.namedWindow("Object detection", cv2.WINDOW_NORMAL) 
        cv2.resizeWindow("Object detection", 700, 700) 
        cv2.imshow('Object detection', frame_with_detections)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Function to detect objects in a frame
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

# Function to visualize detections on a frame
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
        thickness = 2  # Increase for thicker box
        cv2.rectangle(frame_with_detections, start_point, end_point, color, thickness)

        # Draw label and score
        font_scale = 1  # Adjust for font size
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_color = (0, 0, 255)
        cv2.putText(frame_with_detections, label, (box_pixels[1], box_pixels[0] - 20), font, font_scale, text_color, thickness)

    return frame_with_detections

# Run object detection on a video
process_video(0)