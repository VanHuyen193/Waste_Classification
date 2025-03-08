# Python In-built packages
from pathlib import Path
import PIL
import cv2
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration

# External packages
import streamlit as st
from ultralytics import YOLO

# Setting page layout
st.set_page_config(
    page_title="Object Detection using YOLOv8",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Waste Detection using YOLOv8")

# Sidebar
st.sidebar.header("ML Model Config")

# Model Options
model_type = st.sidebar.radio("Select Task", ['Detection'])

confidence = float(st.sidebar.slider("Select Model Confidence", 25, 100, 40)) / 100

# Load Pre-trained ML Model
model_path = "best.pt"

try:
    model = YOLO(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)
    st.stop()

st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio("Select Source", ["Image", "Webcam"])

source_img = None

# If image is selected
if source_radio == "Image":
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", "bmp", "webp")
    )

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                st.warning("⚠️ Please upload an image.")
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image", use_container_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img and st.sidebar.button('Detect Objects'):
            res = model.predict(uploaded_image, conf=confidence)
            res_plotted = res[0].plot()[:, :, ::-1]
            st.image(res_plotted, caption='Detected Image', use_container_width=True)

            try:
                with st.expander("Detection Results"):
                    for box in res[0].boxes:
                        st.write(box.data)
            except Exception as ex:
                st.write("No objects detected.")

# Sidebar for webcam option
elif source_radio == "Webcam":
    st.sidebar.warning("Press 'q' to close the webcam.")

    cap = cv2.VideoCapture(0)  # Open webcam (0: default camera)

    if not cap.isOpened():
        st.error("Could not open webcam.")
    else:
        stframe = st.empty()  # Create a placeholder for video frames

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to grab frame.")
                break

            # Perform object detection
            results = model.predict(frame, conf=confidence)
            detected_frame = results[0].plot()

            # Display the frame
            stframe.image(detected_frame, channels="BGR", use_column_width=True)

            # Stop streaming when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

else:
    st.error("Please select a valid source type!")
