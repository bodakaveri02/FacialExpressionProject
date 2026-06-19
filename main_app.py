import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox
import threading
import time
import webbrowser
import random

# --- Configuration ---
MODEL_PATH = 'emotion_model.h5'
HAARCASCADE_PATH = 'haarcascade_frontalface_default.xml'
# If you don't have haarcascade file, download it from:
# https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml

# Emotion Labels (FER2013 standard)
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
# --- Novel Feature: Well-being Assistant ---
class WellnessAssistant:
    def __init__(self):
        self.last_state = "Neutral"
        self.negative_start_time = None
        self.alert_threshold = 3 # Seconds of negative emotion to trigger alert
        
    def check_mood(self, current_emotion):
        current_time = time.time()
        
        # Define negative emotions
        negative_emotions = ['Angry', 'Sad', 'Fear', 'Disgust']
        
        if current_emotion in negative_emotions:
            if self.last_state == current_emotion:
                # Check if duration exceeds threshold
                if self.negative_start_time and (current_time - self.negative_start_time > self.alert_threshold):
                    self.trigger_intervention(current_emotion)
                    self.negative_start_time = current_time # Reset timer
            else:
                # Start tracking new negative emotion
                self.last_state = current_emotion
                self.negative_start_time = current_time
        else:
            # Reset if emotion changes to positive/neutral
            self.last_state = current_emotion
            self.negative_start_time = None

    def trigger_intervention(self, emotion):
        # Define interventions based on specific negative emotion
        if emotion == 'Sad':
            quote = "It's okay to feel down. Here is something to help."
            link = "https://www.youtube.com/results?search_query=happy+music+playlist"
        elif emotion == 'Angry':
            quote = "Take a deep breath. Count to 10."
            link = "https://www.youtube.com/results?search_query=calming+nature+sounds"
        elif emotion == 'Fear':
            quote = "You are stronger than you think. Courage is taking action."
            link = "https://www.youtube.com/results?search_query=motivational+speech"
        else:
            quote = "Let's change the mood."
            link = "https://www.google.com/search?q=funny+cat+videos"

        print(f"\n--- ALERT: {emotion.upper()} detected for {self.alert_threshold}s ---")
        print(f"Message: {quote}")
        print(f"Opening resource: {link}\n")
        
        # Auto-open browser to help user (Simulated Smart Environment)
        webbrowser.open(link)

# --- Main Application GUI ---
class EmotionDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion-Based Smart Assistant")
        self.root.geometry("800x600")

        # Load Model
        try:
            self.model = load_model(MODEL_PATH)
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + HAARCASCADE_PATH)
        except:
            messagebox.showerror("Error", "Model or Haarcascade file not found. Train the model first!")
            root.destroy()
            return

        self.assistant = WellnessAssistant()
        self.is_running = False

        # GUI Elements
        self.video_label = tk.Label(root)
        self.video_label.pack(pady=10)

        self.emotion_label = tk.Label(root, text="Emotion: Detecting...", font=("Helvetica", 16))
        self.emotion_label.pack(pady=10)

        self.status_label = tk.Label(root, text="Status: Idle", font=("Helvetica", 12), fg="blue")
        self.status_label.pack(pady=5)

        self.start_btn = tk.Button(root, text="Start Detection", command=self.start_video, bg="green", fg="white", font=("Helvetica", 12))
        self.start_btn.pack(pady=5)

        self.stop_btn = tk.Button(root, text="Stop", command=self.stop_video, bg="red", fg="white", font=("Helvetica", 12), state=tk.DISABLED)
        self.stop_btn.pack(pady=5)

    def start_video(self):
        self.is_running = True
        self.cap = cv2.VideoCapture(0)
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Status: Analyzing...")
        self.update_frame()

    def stop_video(self):
        self.is_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Stopped")
        if self.cap:
            self.cap.release()

    def update_frame(self):
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray_frame, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray_frame[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            
            # Preprocess for prediction
            img_pixels = np.expand_dims(roi_gray, axis=0)
            img_pixels = img_pixels / 255.0
            
            predictions = self.model.predict(img_pixels, verbose=0)
            max_index = np.argmax(predictions[0])
            emotion = EMOTIONS[max_index]

            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Update GUI and Assistant
            self.emotion_label.config(text=f"Emotion: {emotion}")
            self.assistant.check_mood(emotion)

        # Convert to Tkinter format
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(10, self.update_frame)

# --- Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionDetectorApp(root)
    root.mainloop()