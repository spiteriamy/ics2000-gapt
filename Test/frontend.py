import tkinter as tk
from tkinter import Button
import cv2
from PIL import Image, ImageTk
import dlib
import numpy as np
from deepface import DeepFace

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

class BasicSnapCam:
    def __init__(self, root):
        self.root = root
        self.root.title("SnapCam")
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open video device.")
            self.root.destroy()
            return

        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        self.root.geometry("640x480")
        self.canvas = tk.Canvas(root, width=640, height=480)
        self.canvas.pack()
        self.btn = Button(root, text="Quit", command=self.quit)
        self.btn.pack(pady=5)
        self.update()

    def quit(self):
        if self.cap:
            self.cap.release()
        self.root.destroy()

    def update(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                face_rect = dlib.rectangle(x, y, x + w, y + h)
                landmarks = predictor(gray, face_rect)
                for i in range(68):
                    pt = (landmarks.part(i).x, landmarks.part(i).y)
                    cv2.circle(frame, pt, 1, (0, 0, 255), -1)

                try:
                    emotion = DeepFace.analyze(frame, actions=['emotion'], detector_backend='skip')[0]['dominant_emotion']
                    cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                except:
                    pass

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
            self.canvas.img = img

        self.root.after(10, self.update)

if __name__ == "__main__":
    root = tk.Tk()
    app = BasicSnapCam(root)
    root.mainloop()
