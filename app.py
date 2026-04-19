import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import cv2
import os
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

# 1. Setup Models with Caching to prevent Segmentation Fault


@st.cache_resource
def load_my_models():
    prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
    weightsPath = os.path.sep.join(
        ["face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
    maskNet = load_model("mask_detector.h5")
    return faceNet, maskNet


faceNet, model = load_my_models()


def detect_and_predict_mask(frame, faceNet, maskNet):
    if frame is None:
        return ([], [])
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces, locs, preds = [], [], []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.4:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            if face is not None and face.size > 0:
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                faces.append(face)
                locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)
    return (locs, preds)


class FaceMaskTransformer(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        (locs, preds) = detect_and_predict_mask(img, faceNet, model)

        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            cv2.putText(img, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(img, (startX, startY), (endX, endY), color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


st.sidebar.title("Settings")
mode = st.sidebar.selectbox("Select Mode", ("Webcam (Live)", "Upload Image"))

if mode == "Webcam (Live)":
    st.title("Live Real-Time Mask Detection")
    webrtc_streamer(
        key="mask-detection",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=FaceMaskTransformer,
        rtc_configuration={"iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
elif mode == "Upload Image":
    st.title("Image Detection")
    file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
    if file:
        img = Image.open(file)
        frame = np.array(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        (locs, preds) = detect_and_predict_mask(frame, faceNet, model)
        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width=700)
