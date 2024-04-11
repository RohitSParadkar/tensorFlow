import cv2
from PIL import Image
import numpy as np
import time
from yolov8 import YOLOv8

# Initialize yolov8 object detector
model_path = "../models/best_v8_latest.onnx"
start = time.time()
yolov8_detector = YOLOv8(model_path, conf_thres=0.1, iou_thres=0.3)


# Read image
img_url = "../output/testImage/threadDamage1.jpg"
img = np.array(Image.open(img_url))
print(type(img))

# Detect Objects
boxes, scores, class_ids = yolov8_detector(img)

# Draw detections

combined_img = yolov8_detector.draw_detections(img)

cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Detected Objects", 600, 600)
cv2.imshow("Detected Objects", combined_img)
cv2.imwrite("doc/img/detected_objects.jpg", combined_img)
end = time.time()
print("The inference is:",(end-start)*1000,"ms")
cv2.waitKey(0)