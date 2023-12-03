from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer
import av

# model = YOLO("model/hard_hat_detection_yolov8_20_epoch.pt")
model = YOLO("model/construction_site_safety_yolov8n_100_epoch.pt")


def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    results = model.predict(source=img)
    img_with_detections = results[0].plot()

    return av.VideoFrame.from_ndarray(img_with_detections, format="bgr24")


webrtc_streamer(key="helmet detector", video_frame_callback=video_frame_callback)