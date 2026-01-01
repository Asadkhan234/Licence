import streamlit as st
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import time
import warnings
warnings.filterwarnings('ignore')


class LicenseDetector:
    def __init__(self, model_path='best.pt', conf_threshold=0.25, iou_threshold=0.45):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.model_path = model_path
        self.confidence_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        try:
            st.info(f"Loading model from {model_path}...")
            self.model = YOLO(model_path)
            self.model.to(self.device)
            st.success(f"Model loaded successfully on {self.device}!")
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            st.warning("Check your ultralytics version matches the version used to train this model.")

    def predict(self, image):
        if self.model is None:
            st.warning("Model not loaded. Cannot run prediction.")
            return None, image

        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Run inference
        try:
            start_time = time.time()
            results = self.model(
                image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                device=self.device
            )
            end_time = time.time()
            # Display timing info
            st.info(f"Inference done in {round((end_time-start_time)*1000, 2)} ms")

            annotated_image = results[0].plot()
            if len(annotated_image.shape) == 3:
                annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

            return results, annotated_image
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return None, image

    def extract_detections(self, results):
        detections = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    detection = {
                        'bbox': box.xyxy[0].cpu().numpy().tolist(),
                        'confidence': float(box.conf[0]),
                        'class_id': int(box.cls[0]),
                        'class_name': results[0].names[int(box.cls[0])]
                    }
                    detections.append(detection)
        return detections


# ----------------- STREAMLIT INTERFACE -----------------

st.title("License Plate Detection")

option = st.radio("Choose input type", ["Image", "Video"])

detector = LicenseDetector("best.pt", conf_threshold=0.25)

if option == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        results, annotated_image = detector.predict(image)

        if annotated_image is not None:
            st.image(annotated_image, caption="Detected Image", use_column_width=True)

            # Download button
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            cv2.imwrite(tmp_file.name, cv2.cvtColor(np.array(annotated_image), cv2.COLOR_RGB2BGR))
            st.download_button("Download Annotated Image", tmp_file.name, file_name="annotated.png")

        if results:
            detections = detector.extract_detections(results)
            st.write("Detections:", detections)

elif option == "Video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        out = cv2.VideoWriter(tmp_out.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        stframe = st.empty()
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_bar = st.progress(0)

        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break

            results, annotated_frame = detector.predict(frame)
            out.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))

            stframe.image(annotated_frame, channels="RGB")
            progress_bar.progress((i+1)/frame_count)

        cap.release()
        out.release()

        st.success("Video processing completed!")
        st.video(tmp_out.name)
        st.download_button("Download Annotated Video", tmp_out.name, file_name="annotated_video.mp4")