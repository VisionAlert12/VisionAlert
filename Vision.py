import streamlit as st
import cv2
import os
import time
from datetime import datetime
from ultralytics import YOLO
import math
import threading
import pyttsx3  
import pygame
import gtts
import uuid


# Text-to-speech library

# Directory to save videos
SAVE_DIR = "recorded_videos"
os.makedirs(SAVE_DIR, exist_ok=True)

# Model setup
model = YOLO("best.pt")

# Object classes
classNames={
            0: 'Bump Ahead', 1: 'Give Way', 2: 'Go Slow', 3: 'Narrow Bridge Ahead', 4: 'No Overtaking', 5: 'No Parking', 6: 'No Uturn', 7: 'School Ahead',
            8: 'Side Road Ahead', 9: 'Speed Limit', 10: 'Stop', 11: 'pedestrian crossing' 
            }
# Initialize Text-to-Speech engine
engine = pyttsx3.init()

# Set the speaking rate (optional)
engine.setProperty('rate', 150)  # Speed of speech (default is 200)
engine.setProperty('volume', 1)  # Volume level (0.0 to 1.0)

# Function for speaking text
def speak(text):
    def run_speech():
        # Generate a unique filename for each speech file
        filename = f"speech_{uuid.uuid4().hex}.mp3"
        tts = gtts.gTTS(text)
        tts.save(filename)
        pygame.mixer.init()
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        os.remove(filename)  # Remove the file after playing
    
    # Run speech in a separate thread to avoid blocking
    speech_thread = threading.Thread(target=run_speech)
    speech_thread.start()

# Streamlit UI setup
st.set_page_config(page_title="Vision Alert", page_icon="🚗", layout="wide")

# Sidebar navigation
st.sidebar.title("Vision Alert Navigation")
option = st.sidebar.radio("Go to:", ("Home", "Live Recording", "Upload Video", "Saved Videos"))
lastl=""
if option == "Home":
    st.title("🚗 Vision Alert")
    st.subheader("Welcome to the Road Sign Detection and Alert System")
    st.markdown(
        """
        - *Analyze road signs in real-time.*
        - *Provide alerts to drivers via live feed and saved videos.*
        - *Navigate to 'Live Recording' to start detection, 'Upload Video' to analyze a video, or 'Saved Videos' to view recordings.*
        """
    )
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
            frame_counter = 0  # Frame counter to reduce frame rate

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.error("⚠ Failed to capture video.")
                    break

                # Only process every 5th frame
                frame_counter += 1
                if frame_counter % 5 != 0:
                    continue

                results = model(frame, stream=True)

                for r in results:
                    boxes = r.boxes

                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        confidence = math.ceil((box.conf[0] * 100)) / 100
                        cls = int(box.cls[0])
                        label = classNames.get(cls, "Unknown")

                        # Speak the label and confidence level
                        speak(f"{label}")

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                        cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

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

        cap = cv2.VideoCapture(video_path)
        video_placeholder = st.empty()
        frame_counter = 0  # Frame counter to reduce frame rate

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Only process every 5th frame
            frame_counter += 1
            if frame_counter % 5 != 0:
                continue

            results = model(frame, stream=True)

            for r in results:
                boxes = r.boxes

                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    confidence = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])
                    label = classNames.get(cls, "Unknown")

                    # Speak the label and confidence level
                    if confidence>0.8:
                        if lastl != label:
                            speak(f"{label}")
                        lastl=label

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame, channels="RGB")

        cap.release()

elif option == "Saved Videos":
    st.title("📁 Saved Videos")

    # Fix: Close the 'endswith' parentheses
    videos = [f for f in os.listdir(SAVE_DIR) if f.endswith((".avi", ".mp4"))]

    if not videos:
        st.info("No saved videos found.")
    else:
        selected_video = st.selectbox("Select a video to play:", videos)

        if selected_video:
            video_path = os.path.join(SAVE_DIR, selected_video)
            st.video(video_path)