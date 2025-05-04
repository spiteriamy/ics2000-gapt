import tkinter as tk
from tkinter import Button, OptionMenu, StringVar, Checkbutton, IntVar
import cv2
from PIL import ImageTk, Image
import dlib
import numpy as np
# from deepface import DeepFace
import time
import threading
from filter_engine import draw_filter
from keras import models

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

emotion_classifier = models.load_model('emotion_model.keras')
emotion_classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

class SnapCam:
    def __init__(self, root):
        self.root = root
        self.root.title("SnapCam")
        self.cap = cv2.VideoCapture(0)
        self.running = True

        self.width = 640
        self.height = 480

        self.canvas = tk.Canvas(root, width=self.width, height=self.height)
        self.canvas.pack()

        control_frame = tk.Frame(root)
        control_frame.pack(pady=5)

        self.filter_var = StringVar(value="auto")
        emotions = ["auto", "happy", "sad", "angry", "fear", "surprise", "disgust"]
        self.filter_menu = OptionMenu(control_frame, self.filter_var, *emotions)
        self.filter_menu.pack(side=tk.LEFT, padx=5)

        self.btn_change_filter = Button(control_frame, text="Next Filter", command=self.force_filter_change)
        self.btn_change_filter.pack(side=tk.LEFT, padx=5)

        self.multi_face_var = IntVar(value=0)
        self.chk_multi = Checkbutton(control_frame, text="Multiple Faces", variable=self.multi_face_var)
        self.chk_multi.pack(side=tk.LEFT, padx=5)

        self.btn_quit = Button(control_frame, text="Quit", command=self.quit)
        self.btn_quit.pack(side=tk.RIGHT, padx=5)

        self.last_emotion_time = 0
        self.last_emotion = "happy"
        self.emotion_lock = threading.Lock()

        self.force_change_filter = False  # flag to force change_interval = 0

        self.update()

    def force_filter_change(self):
        self.force_change_filter = True

    def quit(self):
        if self.cap:
            self.cap.release()
        self.root.destroy()

    def analyze_emotion_async(self, frame):
        def analyze():
            # try:
            #     result = DeepFace.analyze(frame, actions=['emotion'], detector_backend='skip')[0]
            #     with self.emotion_lock:
            #         self.last_emotion = result['dominant_emotion'].lower()
            #         self.last_emotion_time = time.time()
            # except:
            #     pass
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_resized = cv2.resize(gray, (48, 48)).reshape(1, 48, 48, 1) / 255.0
                result = emotion_classifier.predict(gray_resized)
                predicted_emotion = emotion_classes[np.argmax(result)]
                with self.emotion_lock:
                    self.last_emotion = predicted_emotion.lower()
                    self.last_emotion_time = time.time()
            except Exception as e:
                print(f'Failed to analyze emotion: {e}')

        threading.Thread(target=analyze, daemon=True).start()

    def update(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (self.width, self.height))
            frame = cv2.flip(frame, 1)
            small_frame = cv2.resize(frame, (320, 240))
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)

            if not self.multi_face_var.get():
                faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[:1]

            try:
                frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                for (x, y, w, h) in faces:
                    face = dlib.rectangle(x, y, x + w, y + h)
                    landmarks = predictor(gray, face)

                    selected = self.filter_var.get()
                    emotion = selected


                    if selected == "auto":
                        now = time.time()
                        if now - self.last_emotion_time > 10:
                            self.analyze_emotion_async(small_frame)
                        with self.emotion_lock:
                            emotion = self.last_emotion

                    change_interval = 0 if self.force_change_filter else 5
                    frame_pil = draw_filter(frame_pil, landmarks, emotion, (w, h), change_interval)
                    self.force_change_filter = False  # reset after one use

                frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"Filter application error: {e}")

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.canvas.image = imgtk

        self.root.after(10, self.update)

if __name__ == "__main__":
    root = tk.Tk()
    app = SnapCam(root)
    root.mainloop()
