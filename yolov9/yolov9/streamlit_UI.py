import streamlit as st
from PIL import Image
import cv2
import numpy as np
from io import BytesIO
import tempfile
from ultralytics import YOLO
import os
from PIL import Image
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap
from io import BytesIO
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode
import streamlit as st
import cv2
import numpy as np
import av
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.draw.color import ColorPalette
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
from pathlib import Path
import torch
import platform

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

try:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")

# Load the YOLO model
model = YOLO("../../models/best_v8_8th_April.pt")

def predict_and_display_image(image):
    """
    Performs prediction on an image and displays the result.
    """
    # Perform prediction
    results = model.predict(image)
    result = results[0]
    # Convert image from numpy array to PIL Image and resize
    pil_image = Image.fromarray(result.plot()[:,:,::-1])
    desired_width = 600
    desired_height = 600
    pil_image = pil_image.resize((desired_width, desired_height), Image.LANCZOS)
    # Display the image with prediction results
    st.image(pil_image, caption="Predicted image with bounding boxes", use_column_width=True)

def predict_and_display_video(video_bytes):
    """
    Performs prediction on a video file and displays the result.
    """
    # Save video bytes to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".mp4") as temp_file:
        temp_file.write(video_bytes)
        temp_file.seek(0)
        # Read the video from the temporary file
        cap = cv2.VideoCapture(temp_file.name)
        frame_placeholder = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame to 600x600
            frame = cv2.resize(frame, (600, 600))

            # Convert frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Perform prediction on the frame
            results = model.predict(frame_rgb, conf=0.3)
            result = results[0]

            # Display the frame with prediction results
            pil_image = Image.fromarray(result.plot()[:,:,::-1])
            pil_image = pil_image.resize((600, 600), Image.LANCZOS)
            frame_placeholder.image(pil_image, caption="Predicted frame with bounding boxes", use_column_width=True)

        # Release the video capture object
        cap.release()

def run(
        weights="../../models/best_v9_26_Mar.pt",  # model path or triton URL
        source="../../test.jpg",  # file/dir/URL/glob/screen/0(webcam)
        data='data/coco.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=True,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project='runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=10,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    is_video = source.endswith(('.mp4', '.avi', '.mov'))
    is_image = source.endswith(('.png', '.jpg', '.jpeg', '.gif'))
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    # Button to stop webcam
    stop_webcam_button = st.button("Stop Webcam") if not (is_image or is_video) else None
    frame_placeholder = st.empty()
    stop_webcam = False  # Initialize stop flag
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            thickness = line_thickness if is_image else 3
            print("=>>>>>>>>>>>>", is_image)
            annotator = Annotator(im0, line_width=thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)


            if stop_webcam_button:  # Check if stop button is pressed
                stop_webcam = True  # Set stop flag
                return  # Break out of the loop

            # Stream results
            im0 = annotator.result()
            if is_image:
                st.image(im0, caption="Predicted image with bounding boxes", use_column_width=True)
            elif is_video:
                frame_placeholder.image(im0, caption="Predicted frame with bounding boxes", use_column_width=True)
            else:
                frame_placeholder.image(im0, caption="Predicted frame with bounding boxes", use_column_width=True)


        if stop_webcam:
            # Clear frame placeholder and break loop if webcam stopped
            frame_placeholder.empty()
            return

        # Print results
        t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)

box_annotator = BoxAnnotator(color=ColorPalette(), thickness=3, text_thickness=1, text_scale=0.5)

CLASS_NAMES_DICT = model.model.names

def predict(frame):
    results = model.predict(frame, conf=0.1)
    return results

def plot_bboxes(results, frame):
    xyxys = []
    confidences = []
    class_ids = []
    # Extract detections for person class
    for result in results[0]:
        class_id = result.boxes.cls.cpu().numpy().astype(int)
        if class_id == 0:
            xyxys.append(result.boxes.xyxy.cpu().numpy())
            confidences.append(result.boxes.conf.cpu().numpy())
            class_ids.append(result.boxes.cls.cpu().numpy().astype(int))
    # Setup detections for visualization
    detections = Detections(
                xyxy=results[0].boxes.xyxy.cpu().numpy(),
                confidence=results[0].boxes.conf.cpu().numpy(),
                class_id=results[0].boxes.cls.cpu().numpy().astype(int),
                )
    # Format custom labels
    labels = [f"{CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
    for _, confidence, class_id, tracker_id
    in detections]
    # Annotate and display frame
    frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)
    frame = np.fliplr(frame)
    return frame

class VideoProcessorv8:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = predict(img)
        img = plot_bboxes(results,img)
        img = np.fliplr(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Title and description
st.title("YOLOv8 Object Detection App")
st.write("Upload an image or a video and see the object detection results using YOLO model.")

# Model Version Selection
modelType = 'Yolov8'

if (modelType != 'None'):
    inputType = st.selectbox('Select the input type', ('None', 'Image or Video', 'Web Cam'))
    st.write('You selected:', inputType)

    if (inputType != 'None'):

        if (inputType == 'Image or Video'):
            # Upload section
            uploaded_file = st.file_uploader("Upload image or video", type=["jpg", "png", "mp4"])

            if uploaded_file is not None:
                if modelType == 'Yolov8':
                    if uploaded_file.type.startswith('image'):
                        # If uploaded file is an image
                        image = Image.open(uploaded_file)
                        predict_and_display_image(image)
                    elif uploaded_file.type.startswith('video'):
                        # If uploaded file is a video
                        video_bytes = uploaded_file.read()
                        predict_and_display_video(video_bytes)
                elif modelType == 'Yolov9':
                    if uploaded_file.type.startswith('image'):
                        # If uploaded file is an image
                        image = Image.open(uploaded_file)

                        # Save the uploaded image to a temporary file
                        image_file_path = "uploaded_image.jpg"  # Change the file name as needed
                        image.save(image_file_path)

                        # Call the run function with the file path
                        run(source=image_file_path)
                        # st.image(result_image, caption="Predicted image with bounding boxes", use_column_width=True)

                        # Delete the file at the end of the conditional block
                        os.remove(image_file_path)    

                    elif uploaded_file.type.startswith('video'):
                        # If uploaded file is a video
                        video_bytes = uploaded_file.read()
                        # Save the uploaded video to a file in the same directory
                        video_file_path = "uploaded_video.mp4"  # Change the file name as needed
                        with open(video_file_path, "wb") as video_file:
                            video_file.write(video_bytes)
                        # Call the run function with the file path
                        run(source=video_file_path)

                        # Delete the file at the end of the conditional block
                        os.remove(video_file_path)
        elif (inputType == 'Web Cam'):
            st.header("Webcam Live Feed")
            st.write("Click on start to use webcam and detect your face emotion")
            if modelType == 'Yolov8':
                webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION, media_stream_constraints={"video": True, "audio": False}, video_processor_factory=VideoProcessorv8, async_processing=True )
            else:
                run(source='0')