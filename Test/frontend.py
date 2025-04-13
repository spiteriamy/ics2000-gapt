import tkinter as tk
from tkinter import Button, OptionMenu, StringVar, Checkbutton, IntVar
import cv2
from PIL import Image, ImageTk
import dlib
import numpy as np
from deepface import DeepFace
import os
import glob
import time
import threading

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

class SnapCam:
    def __init__(self, root):
        self.root = root
        self.root.title("SnapCam")
        self.cap = cv2.VideoCapture(0)
        self.running = True

        self.width = 960
        self.height = 720

        self.canvas = tk.Canvas(root, width=self.width, height=self.height)
        self.canvas.pack()

        control_frame = tk.Frame(root)
        control_frame.pack(pady=5)

        self.filter_var = StringVar(value="auto")
        emotions = ["auto", "happy", "sad", "angry", "fear", "surprised", "disgust"]
        self.filter_menu = OptionMenu(control_frame, self.filter_var, *emotions)
        self.filter_menu.pack(side=tk.LEFT, padx=5)

        self.btn_next_filter = Button(control_frame, text="Next Filter", command=self.next_filter)
        self.btn_next_filter.pack(side=tk.LEFT, padx=5)

        self.multi_face_var = IntVar(value=0)
        self.chk_multi = Checkbutton(control_frame, text="Multiple Faces", variable=self.multi_face_var)
        self.chk_multi.pack(side=tk.LEFT, padx=5)

        self.btn_quit = Button(control_frame, text="Quit", command=self.quit)
        self.btn_quit.pack(side=tk.RIGHT, padx=5)

        self.current_filter_index = 0
        self.filter_cache = {}
        self.last_emotion_time = 0
        self.last_emotion = "happy"
        self.emotion_lock = threading.Lock()

        self.update()

    def quit(self):
        if self.cap:
            self.cap.release()
        self.root.destroy()

    def overlay_filter(self, frame, filter_img, x, y, w, h):
        try:
            overlay = cv2.imread(filter_img, cv2.IMREAD_UNCHANGED)
            if overlay is None:
                return
            overlay = cv2.resize(overlay, (w, h))
            for i in range(h):
                for j in range(w):
                    if y + i >= frame.shape[0] or x + j >= frame.shape[1] or x + j < 0 or y + i < 0:
                        continue
                    alpha = overlay[i, j, 3] / 255.0
                    for c in range(3):
                        frame[y + i, x + j, c] = (1 - alpha) * frame[y + i, x + j, c] + alpha * overlay[i, j, c]
        except:
            pass

    def get_filter_paths(self, emotion):
        if emotion not in self.filter_cache:
            pattern = os.path.join("Filters", f"{emotion}*.png")
            self.filter_cache[emotion] = sorted(glob.glob(pattern))
        return self.filter_cache[emotion]

    def next_filter(self):
        emotion = self.filter_var.get()
        if emotion == "auto":
            emotion = self.last_emotion
        filters = self.get_filter_paths(emotion)
        if filters:
            self.current_filter_index = (self.current_filter_index + 1) % len(filters)

    def analyze_emotion_async(self, frame):
        def analyze():
            try:
                result = DeepFace.analyze(frame, actions=['emotion'], detector_backend='skip')[0]
                with self.emotion_lock:
                    self.last_emotion = result['dominant_emotion'].lower()
                    self.last_emotion_time = time.time()
            except:
                pass
        threading.Thread(target=analyze, daemon=True).start()

    def update(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (self.width, self.height))
            frame = cv2.flip(frame, 1)
            small_frame = cv2.resize(frame, (320, 240))  # Faster processing
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)

            # Scale face coordinates to match full frame
            scale_x = self.width / 320
            scale_y = self.height / 240
            faces = [(int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y)) for (x, y, w, h) in faces]

            if not self.multi_face_var.get():
                faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[:1]

            for (x, y, w, h) in faces:
                face_rect = dlib.rectangle(x, y, x + w, y + h)
                landmarks = predictor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), face_rect)

                selected = self.filter_var.get()
                emotion = selected

                if selected == "auto":
                    now = time.time()
                    if now - self.last_emotion_time > 5:
                        self.analyze_emotion_async(small_frame)
                    with self.emotion_lock:
                        emotion = self.last_emotion
                    cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

                filters = self.get_filter_paths(emotion)
                if filters:
                    index = self.current_filter_index % len(filters)
                    filter_file = filters[index]
                    if os.path.basename(filter_file).lower() == "angry2.png":
                        filter_y = y - int(h * 1.2)
                    else:
                        filter_y = y - int(h * 0.5)
                    self.overlay_filter(frame, filter_file, x - int(w * 0.25), filter_y, int(w * 1.5), int(h * 2))

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
            self.canvas.img = img

        self.root.after(1, self.update)

if __name__ == "__main__":
    root = tk.Tk()
    app = SnapCam(root)
    root.mainloop()
