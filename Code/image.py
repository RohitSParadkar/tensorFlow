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

# PROVIDE PATH TO IMAGE DIRECTORY
IMAGE_PATHS = '../output/testImage/threadDamage3.jpg'

# PROVIDE PATH TO MODEL DIRECTORY
PATH_TO_MODEL_DIR = '../Code/content/exported_models/ssd_20ks_model'

# PROVIDE PATH TO LABEL MAP
PATH_TO_LABELS = '../models/Top-Damage_label_map.pbtxt'

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

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.
    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.
    Args:
      path: the file path to the image
    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))

print('Running inference for {}... '.format(IMAGE_PATHS), end='')
image = load_image_into_numpy_array(IMAGE_PATHS)

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
input_tensor = input_tensor[tf.newaxis, ...]

detections = detect_fn(input_tensor)

num_detections = int(detections.pop('num_detections'))
detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
detections['num_detections'] = num_detections

# detection_classes should be ints.
detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
# print(detections['detection_classes'])
# print(detections['detection_scores'])

image_with_detections = image.copy()

# Filter out bounding boxes with confidence scores below MIN_CONF_THRESH
filtered_boxes = detections['detection_boxes'][detections['detection_scores'] >= MIN_CONF_THRESH]
filtered_classes = detections['detection_classes'][detections['detection_scores'] >= MIN_CONF_THRESH]
filtered_scores = detections['detection_scores'][detections['detection_scores'] >= MIN_CONF_THRESH]
print("filter box",filtered_boxes.shape)
print("filter class",filtered_classes.shape)
print("filter score",filtered_scores.shape)
# Apply Non-Maximum Suppression
selected_indices = tf.image.non_max_suppression(filtered_boxes, filtered_scores, max_output_size=50, iou_threshold=0.5)  # iou_threshold (overlaping area of bounding box 50% )
selected_boxes = tf.gather(filtered_boxes, selected_indices)
selected_classes = tf.gather(filtered_classes, selected_indices)
selected_scores = tf.gather(filtered_scores, selected_indices)


# Visualization of the results of a detection.
for i in range(selected_boxes.shape[0]):
  class_id = int(selected_classes[i])
  score = selected_scores[i].numpy()
  label = '{}: {:.2f}'.format(category_index[class_id]['name'], score)
  print("\nlabels are:",label)

  # Convert box to pixel coordinates
  height, width, _ = image_with_detections.shape
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
  cv2.rectangle(image_with_detections, start_point, end_point, color, thickness)

  # Draw label and score
  font_scale = 2  # Adjust for font size
  font = cv2.FONT_HERSHEY_SIMPLEX
  text_color = (0, 0, 255)
  cv2.putText(image_with_detections, label, (box_pixels[1], box_pixels[0] - 20), font, font_scale, text_color, thickness)

cv2.namedWindow("Object detection", cv2.WINDOW_NORMAL) 
cv2.resizeWindow("Object detection", 600, 600)
cv2.imshow('Object detection', image_with_detections)
cv2.waitKey(0)
cv2.destroyAllWindows()

