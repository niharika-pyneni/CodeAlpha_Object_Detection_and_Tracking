import streamlit as st
import cv2
import time
from ultralytics import YOLO


st.set_page_config(
    page_title="AI Object Detection",
    page_icon="🎯"
)


st.title("🎯 AI Object Detection and Tracking")

st.write(
    "Real-time object detection and tracking using YOLO + OpenCV"
)


# Better accuracy model
model = YOLO("yolov8s.pt")


source = st.radio(
    "Choose input",
    ["Webcam", "Video File"]
)


confidence = st.slider(
    "Detection Confidence",
    0.1,
    1.0,
    0.5
)


if source == "Webcam":

    start = st.checkbox(
        "Start Camera"
    )


    FRAME_WINDOW = st.image([])

    fps_text = st.empty()


    camera = cv2.VideoCapture(0)


    previous_time = 0


    while start:

        ret, frame = camera.read()

        if not ret:
            st.error("Camera not found")
            break


        results = model.track(
            frame,
            persist=True,
            conf=confidence
        )


        output = results[0].plot()


        current_time = time.time()

        fps = 1 / (current_time - previous_time)

        previous_time = current_time


        cv2.putText(
            output,
            f"FPS: {int(fps)}",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )


        FRAME_WINDOW.image(
            output,
            channels="BGR"
        )


    camera.release()



else:

    uploaded_file = st.file_uploader(
        "Upload video",
        type=["mp4","avi"]
    )


    if uploaded_file:

        with open(
            "input.mp4",
            "wb"
        ) as f:

            f.write(
                uploaded_file.read()
            )


        video = cv2.VideoCapture(
            "input.mp4"
        )


        FRAME_WINDOW = st.image([])


        while video.isOpened():

            ret, frame = video.read()


            if not ret:
                break


            results = model.track(
                frame,
                persist=True,
                conf=confidence
            )
            output = results[0].plot()
            object_count = len(results[0].boxes)

            cv2.putText(
                output,
                f"Objects: {object_count}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            FRAME_WINDOW.image(
                output,
                channels="BGR"
            )
        video.release()