import cv2
from deepface import DeepFace

def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        image = cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 2)
        try:
            # analyze emotion and display
            analyze = DeepFace.analyze(vid, actions=['emotion'], detector_backend='skip')
            cv2.putText(image, analyze[0]["dominant_emotion"], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            print(analyze[0]["dominant_emotion"])
        except Exception as e:
            print(f'Error: {e}')
    return faces


# loading haar cascade classifier
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


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

    # apply bounding box to the video frame
    faces = detect_bounding_box(
        flipped_frame
    )  

    # Display the resulting frame
    cv2.imshow('Webcam Feed', flipped_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()