import os
import cv2
import numpy as np
import keras
from keras import utils, callbacks
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from emotionclassifier import model


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


# loading the training data

dataset_path = os.path.join('data', 'train')
images_train, labels_train = load_data(dataset_path)

# split into training and validation

images_train, images_val, labels_train, labels_val = train_test_split(
    images_train, labels_train, test_size=0.2, random_state=42
)

# load test data
dataset_test_path = os.path.join('data', 'test')
images_test, labels_test = load_data(dataset_test_path)

print(f"Training data: {len(images_train)} images, {len(labels_train)} labels")
print(f"Validation data: {len(images_val)} images, {len(labels_val)} labels")
print(f"Testing data: {len(images_test)} images, {len(labels_test)} labels")


# TRAINING

# early stopping to stop training when val loss stops imporoving
early_stopping = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,  # Stop after 10 epochs without improvement
    restore_best_weights=True
)

history = model.fit(
    images_train, 
    labels_train, 
    validation_data=(images_val, labels_val),
    epochs=100,
    batch_size=64,
    callbacks=[early_stopping] # early stopping
)

# plotting accuracy

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training & validation loss curves
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

test_loss, test_accuracy = model.evaluate(images_test, labels_test)
print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")

# save model

model.save("emotion_model.keras")
