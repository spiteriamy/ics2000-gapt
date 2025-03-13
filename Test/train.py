import os
import cv2
import numpy as np
from keras import utils, callbacks, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from emotionclassifier import model
import tensorflow as tf


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


# for use with transfer learning model not custom model
# def resize_image(image, label):
#     image = tf.image.grayscale_to_rgb(image)
#     image = tf.image.resize(image, (224, 224))  # Resize dynamically
#     return image, label


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


# class weighting because of dataset imbalamce
# CLASS WEIGHTS NOT WORKING PROPERLY
# CAUSING SIGNIFICANT ISSUES TO PERFORMANCE
emotion_counts = np.sum(labels_train, axis=0)  # Count samples per class
print("Samples per class:", emotion_counts)
# class_weights = compute_class_weight(
#     'balanced', 
#     classes=np.arange(len(emotion_counts)), 
#     y=np.argmax(labels_train, axis=1))
# class_weights = dict(enumerate(class_weights))


# Data augmentation

# datagen = preprocessing.image.ImageDataGenerator(
#     rescale=1./255
#     rotation_range=20,  # More variation
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True
# )
# datagen.fit(images_train)

# for use with transfer learning model not custom model:
# train_dataset = tf.data.Dataset.from_tensor_slices((images_train, labels_train))
# # Apply resizing & batching
# AUTOTUNE = tf.data.AUTOTUNE
# train_dataset = (
#     train_dataset
#     .map(resize_image, num_parallel_calls=AUTOTUNE)  # Resize images
#     .batch(64)  # Set batch size
#     .prefetch(AUTOTUNE)  # Improve performance
# )
# # Convert validation & test data into tf.data.Dataset
# val_dataset = tf.data.Dataset.from_tensor_slices((images_val, labels_val)).map(resize_image).batch(64)
# test_dataset = tf.data.Dataset.from_tensor_slices((images_test, labels_test)).map(resize_image).batch(64)


# TRAINING

# early stopping to stop training when val accuracy stops imporoving
early_stopping = callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=7,  # Stop after 7 epochs without improvement
    restore_best_weights=True
)

history = model.fit(
    images_train, 
    labels_train, 
    # train_dataset,
    validation_data=(images_val, labels_val),
    # validation_data=val_dataset,
    epochs=30,
    batch_size=128,
    # class_weight=class_weights  # Apply class weights
    # callbacks=[early_stopping] # early stopping
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
# test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")

# save model

model.save("emotion_model.keras")
