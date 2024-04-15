import cv2
from yolov8 import YOLOv8
from threading import Thread, Lock
import time

class YOLOv8Threaded:
    def __init__(self, video_source, model_path, window_name):
        self.cap = cv2.VideoCapture(video_source)
        self.window_name = window_name
        self.yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)
        self.thread = Thread(target=self.process_frames)
        self.thread.daemon = True
        self.stop_flag = False
        self.frame = None
        self.frame_lock = Lock()

    def process_frames(self):
        while not self.stop_flag:
            ret, frame = self.cap.read()
            if not ret:
                break

            with self.frame_lock:
                self.frame = frame.copy()

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_flag = True
        self.thread.join()
        self.cap.release()

    def get_latest_frame(self):
        with self.frame_lock:
            return self.frame

if __name__ == "__main__":
    webcam_sources = [0, 1]  # Adjust based on your webcam indices
    model_path = "../../models/best_10_04_2023.onnx"
    window_names = [f"Detected Objects {idx}" for idx in range(len(webcam_sources))]
    
    yolo_threads = []
    for idx, source in enumerate(webcam_sources):
        yolo_thread = YOLOv8Threaded(source, model_path, window_names[idx])
        yolo_thread.start()
        yolo_threads.append(yolo_thread)

    while True:
        start_time = time.time()

        for idx, yolo_thread in enumerate(yolo_threads):
            frame = yolo_thread.get_latest_frame()
            if frame is not None:
                boxes, scores, class_ids = yolo_thread.yolov8_detector(frame)
                combined_img = yolo_thread.yolov8_detector.draw_detections(frame)
                cv2.imshow(window_names[idx], combined_img)

        if cv2.waitKey(1) == ord('q'):
            break

        # Wait to sync processing with FPS
        elapsed_time = time.time() - start_time
        if elapsed_time < 1 / 30:  # Adjust according to desired FPS
            time.sleep(1 / 30 - elapsed_time)

    for yolo_thread in yolo_threads:
        yolo_thread.stop()

    cv2.destroyAllWindows()
