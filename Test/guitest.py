import tkinter as tk
from tkinter import Button, filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from deepface import DeepFace

class WebcamApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        # OpenCV video capture
        self.cap = None

        # loading haar cascade classifier
        self.face_classifier = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # canvas to display the webcam feed
        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.pack()

        # Button to start/stop the webcam
        self.btn_toggle = Button(window, text="Start Camera", width=15, command=self.toggle_camera)
        self.btn_toggle.pack(pady=10)
        self.camera_on = False # track if cam is on or off

        # Button to upload an image
        self.btn_upload = Button(window, text="Upload Image", width=15, command=self.upload_image)
        self.btn_upload.pack(pady=10)

        # default black background for when cam is off
        blank_image = Image.new("RGB", (640, 480), (0, 0, 0))
        self.blank_photo = ImageTk.PhotoImage(image=blank_image)
        self.canvas.create_image(0, 0, image=self.blank_photo, anchor=tk.NW) # display

        # Start the GUI event loop
        self.window.mainloop()

    def toggle_camera(self):
        if self.camera_on:
            # Stop the camera
            self.camera_on = False
            self.btn_toggle.config(text="Start Camera")
            if self.cap:
                self.cap.release()
                self.cap = None

            # Clear the canvas to remove the cam image
            self.canvas.delete("all") 
            self.canvas.create_image(0, 0, image=self.blank_photo, anchor=tk.NW)
        else:
            # Start the camera
            self.camera_on = True
            self.btn_toggle.config(text="Stop Camera")
            self.cap = cv2.VideoCapture(0)
            self.show_frame()

    def show_frame(self):
        if self.camera_on:
            # Capture frame-by-frame
            ret, frame = self.cap.read()

            if ret:
                # Flip the frame horizontally
                flipped_frame = cv2.flip(frame, 1)

                # apply bounding box to the video frame
                faces = self.detect_face(flipped_frame)  

                # Convert the frame from BGR to RGB
                flipped_frame = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB)

                # Resize the frame to fit the canvas
                flipped_frame = cv2.resize(flipped_frame, (640, 480))

                # Convert the frame to a PhotoImage object
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(flipped_frame))

                # Display the frame on the canvas
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

            # Repeat the process after 10 milliseconds
            self.window.after(10, self.show_frame)

    def upload_image(self):
        # Stop the camera if it's running
        if self.camera_on:
            self.toggle_camera()

        # Open a file dialog to select an image file
        file_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif")]
        )

        if file_path:
            # Load the selected image using Pillow
            pil_image = Image.open(file_path)

            # Convert the Pillow image to a NumPy array
            np_image = np.array(pil_image)

            # Convert RGB to BGR (OpenCV uses BGR format)
            np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)

            # apply bounding box to image
            faces = self.detect_face(np_image) 

            # Convert the frame from BGR to RGB
            np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)

            # Convert the NumPy array back to a Pillow image
            pil_image = Image.fromarray(np_image)

            # Resize the image to fit the canvas
            target_width, target_height = 640, 480
            original_width, original_height = pil_image.size

            original_aspect_ratio = original_width / original_height
            target_aspect_ratio = target_width / target_height

            if original_aspect_ratio > target_aspect_ratio:
                # Image is wider than the target size
                new_width = target_width
                new_height = int(target_width / original_aspect_ratio)
            else:
                # Image is taller than the target size
                new_height = target_height
                new_width = int(target_height * original_aspect_ratio)
            
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Create a new image with the target size and black background
            padded_image = Image.new("RGB", (target_width, target_height), (0, 0, 0))

            # Calculate the position to paste the resized image (centered)
            paste_x = (target_width - new_width) // 2
            paste_y = (target_height - new_height) // 2

            # Paste the resized image onto the padded image
            padded_image.paste(pil_image, (paste_x, paste_y))

            # Convert the Pillow image to a Tkinter PhotoImage
            self.uploaded_photo = ImageTk.PhotoImage(image=padded_image)

            # Display the uploaded image on the canvas
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, image=self.uploaded_photo, anchor=tk.NW)

    def detect_face(self, frame):
        # gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # faces = self.face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
        # for (x, y, w, h) in faces:
        #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # return faces
    
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
        for (x, y, w, h) in faces:
            image = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            try:
                # analyze emotion and display
                analyze = DeepFace.analyze(frame, actions=['emotion'], detector_backend='skip')
                cv2.putText(image, analyze[0]["dominant_emotion"], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                print(analyze[0]["dominant_emotion"])
            except Exception as e:
                print(f'Error: {e}')
        return faces

    def __del__(self):
        if self.cap:
            self.cap.release()


# Create the Tkinter window and pass it to the WebcamApp class
root = tk.Tk()
app = WebcamApp(root, "Webcam GUI Application")
