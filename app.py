import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import cv2
import os
from PIL import Image

# 1. Setup Models
prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
weightsPath = os.path.sep.join(
    ["face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
model = load_model("mask_detector.h5")


def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.4:  # Lowered threshold to help detection
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            if face.any():
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


# --- STREAMLIT UI ---
st.sidebar.title("Settings")
choice = st.sidebar.selectbox(
    "Select Mode", ("Webcam", "Upload Image", "Upload Video"))

if choice == "Webcam":
    st.title("Live Webcam Detection")
    run = st.checkbox('Start Webcam')
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    while run:
        _, frame = camera.read()
        frame = cv2.flip(frame, 1)  # Mirror view
        (locs, preds) = detect_and_predict_mask(frame, faceNet, model)

        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (255, 0, 0)
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    else:
        st.write("Webcam Stopped")

elif choice == "Upload Image":
    st.title("Image Detection")
    file = st.file_uploader("Upload an Image", type=['jpg', 'png', 'jpeg'])
    if file:
        image = Image.open(file)
        frame = np.array(image)
        (locs, preds) = detect_and_predict_mask(frame, faceNet, model)

        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (255, 0, 0)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        st.image(frame, use_column_width=True)

elif choice == "Upload Video":
    st.title("Video Detection")
    st.info("Upload a short clip to process.")
    video_file = st.file_uploader("Upload Video", type=['mp4', 'mov', 'avi'])
    if video_file:
        # Saving temp file to process
        with open("temp_video.mp4", "wb") as f:
            f.write(video_file.read())

        cap = cv2.VideoCapture("temp_video.mp4")
        st_frame = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            (locs, preds) = detect_and_predict_mask(frame, faceNet, model)
            for (box, pred) in zip(locs, preds):
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred
                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (255, 0, 0)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            st_frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
