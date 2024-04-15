import torch
from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.torch_utils import select_device, smart_inference_mode
from utils.augmentations import letterbox
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode
from utils.plots import Annotator, colors, save_one_box
import streamlit as st

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

@smart_inference_mode()
def predict(image, imgsz=640, conf_thres=0.1, iou_thres=0.45):
    device = select_device('0')
    model = DetectMultiBackend(weights='../../models/best_v9.pt', device="0", fp16=False, data='data/coco.yaml')
    stride, names, pt = model.stride, model.names, model.pt

    img = letterbox(image, imgsz, stride=stride, auto=True)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img, augment=False, visualize=False)
    pred = non_max_suppression(pred[0][0], conf_thres, iou_thres, classes=None, max_det=1000)

    # Convert predictions to image with bounding boxes
    annotated_image = Annotator(image, line_width=3, text_size=2)
    annotated_image.add(pred)
    annotated_image = annotated_image.render()

    return annotated_image

class VideoProcessorv8(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = predict(image=img)
        return img

st.header("Webcam Live Feed")
st.write("Click on start to use webcam and detect your face emotion")
webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION, media_stream_constraints={"video": True, "audio": False}, video_processor_factory=VideoProcessorv8, async_processing=True)
