import cv2
import dlib
import numpy as np
from deepface import DeepFace
from keras import models

# load pre-trained model for landmark detection (dlib's 68-point landmark model)
predictor_path = "shape_predictor_68_face_landmarks.dat"

# initialise dlib's landmark predictor
predictor = dlib.shape_predictor(predictor_path)

# load haar cascade classifier for face detection
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# load emotion classifier model
# emotion_classifier = models.load_model('emotion_model.h5')
emotion_classifier = models.load_model('emotion_model.keras')
emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


# detect the faces in front of the camera
def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))

    for (x, y, w, h) in faces:
        # draw the bounding box around the face
        image = cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # convert bounding box to a dlib rectangle (landmark predictor will use it)
        face = dlib.rectangle(x, y, x + w, y + h)

        # detect landmarks for a face
        landmarks = predictor(gray_image, face)

        # draw landmarks on the face
        draw_landmarks(vid, landmarks)

        # Analyze emotion and display it

        # using deepface library:
        # try:
        #     analyze = DeepFace.analyze(vid, actions=['emotion'], detector_backend='skip')
        #     cv2.putText(image, analyze[0]["dominant_emotion"], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        #     print(analyze[0]["dominant_emotion"])
        # except Exception as e:
        #     print(f'Error: {e}')

        # analyzing emotion with new model
        face = cv2.resize(gray_image, (48, 48)).reshape(1, 48, 48, 1) / 255.0
        prediction = emotion_classifier.predict(face)
        emotion = emotions[np.argmax(prediction)]
        cv2.putText(vid, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return faces


# function to draw the 68 detected landmarks on the face
def draw_landmarks(vid, landmarks):
    for i in range(68):
        x_landmark = landmarks.part(i).x
        y_landmark = landmarks.part(i).y
        cv2.circle(vid, (x_landmark, y_landmark), 1, (0, 0, 255), -1)


# Ask for webcam access
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Loop to capture and display video frames
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If frame is read correctly, ret is True
    if not ret:
        print("Error: Could not read frame.")
        break

    # Flip the frame horizontally
    flipped_frame = cv2.flip(frame, 1)

    # Apply bounding box and landmarks to the video frame
    faces = detect_bounding_box(flipped_frame)

    # Display the resulting frame
    cv2.imshow('Webcam Feed', flipped_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
