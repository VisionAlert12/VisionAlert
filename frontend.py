import streamlit as st
import cv2
import os
from vision_alert import process_frame, draw_detections, SAVE_DIR # type: ignore

st.set_page_config(page_title="Vision Alert", page_icon="🚗", layout="wide")
st.sidebar.title("Vision Alert Navigation")
option = st.sidebar.radio("Go to:", ("Home", "Live Recording", "Upload Video", "Saved Videos"))

if option == "Home":
    st.title("🚗 Vision Alert")
    st.subheader("Welcome to the Road Sign Detection and Alert System")
    st.image("https://via.placeholder.com/800x400?text=Vision+Alert", use_column_width=True)

elif option == "Live Recording":
    st.title("🎥 Live Recording")
    start_button = st.button("Start Live Detection")
    stop_button = st.button("Stop Live Detection")
    video_placeholder = st.empty()

    if start_button:
        cap = cv2.VideoCapture(0)
        cap.set(3, 640)
        cap.set(4, 480)
        
        if not cap.isOpened():
            st.error("⚠ Unable to access the camera.")
        else:
            st.info("Detection started. Press 'Stop Live Detection' to end.")
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.error("⚠ Failed to capture video.")
                    break
                detections = process_frame(frame)
                frame = draw_detections(frame, detections)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame, channels="RGB")
                if stop_button:
                    break
            cap.release()

elif option == "Upload Video":
    st.title("📤 Upload and Analyze Video")
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        video_path = os.path.join(SAVE_DIR, uploaded_file.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Video uploaded successfully: {uploaded_file.name}")

elif option == "Saved Videos":
    st.title("📁 Saved Videos")
    videos = [f for f in os.listdir(SAVE_DIR) if f.endswith((".avi", ".mp4"))]
    if not videos:
        st.info("No saved videos found.")
    else:
        selected_video = st.selectbox("Select a video to play:", videos)
        if selected_video:
            st.video(os.path.join(SAVE_DIR, selected_video))
