import cv2
from yolov8 import YOLOv8

# # Initialize video
# cap = cv2.VideoCapture("input.mp4")

video_path = '../../output/testVideo/test4.mp4'
cap = cv2.VideoCapture(video_path)
start_time = 5 # skip first {start_time} seconds
cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * cap.get(cv2.CAP_PROP_FPS))

# out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), cap.get(cv2.CAP_PROP_FPS), (3840, 2160))

# Initialize YOLOv7 model
model_path = "../../models/best_10_04_2023.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
while cap.isOpened():

    # Press key q to stop
    if cv2.waitKey(1) == ord('q'):
        break

    try:
        # Read frame from the video
        ret, frame = cap.read()
        if not ret:
            break
    except Exception as e:
        print(e)
        continue

    # Update object localizer
    boxes, scores, class_ids = yolov8_detector(frame)

    combined_img = yolov8_detector.draw_detections(frame)
    cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Detected Objects", 800, 800)
    cv2.imshow("Detected Objects", combined_img)
    # out.write(combined_img)

# out.release()
