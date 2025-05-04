from keras import models, utils
import os
import cv2
import numpy as np

def load_data(directory):
    # emotion mapping
    emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

    images = []
    labels = [] # will map image to emotion

    for emotion_idx, emotion in enumerate(emotions):
        emotion_path = os.path.join(directory, emotion) # construct full path for emotion

        for img_name in os.listdir(emotion_path):
            img_path = os.path.join(emotion_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue # skip unreadable images
            img = cv2.resize(img, (48, 48))
            images.append(img)
            labels.append(emotion_idx) # numerical label

    # convert to numpy array
    images = np.array(images).reshape(-1, 48, 48, 1) / 255.0  # Normalize
    labels = utils.to_categorical(labels, num_classes=len(emotions)) # one hot encoding

    return images, labels

dataset_test_path = os.path.join('data', 'test')
images_test, labels_test = load_data(dataset_test_path)

emotion_classifier = models.load_model('emotion_model.keras')

emotion_classifier.summary()

loss, accuracy = emotion_classifier.evaluate(images_test, labels_test)
print(f"Accuracy: {accuracy * 100:.2f}%")
