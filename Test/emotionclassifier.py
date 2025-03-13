from keras import layers, models, applications
import keras

input_shape = (48, 48, 1)
output_class = 7

model = models.Sequential()

# convolutional layers
model.add(layers.Conv2D(128, kernel_size=(3,3), activation='relu', input_shape=input_shape))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Dropout(0.4))

model.add(layers.Conv2D(256, kernel_size=(3,3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Dropout(0.4))

model.add(layers.Conv2D(512, kernel_size=(3,3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Dropout(0.4))

model.add(layers.Conv2D(512, kernel_size=(3,3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Dropout(0.4))

model.add(layers.Flatten())

# fully connected layers
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.3))

# output layer
model.add(layers.Dense(output_class, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()


# Transfer Learning using a pre trained model

# base_model = applications.MobileNetV2()
# base_model.trainable = False

# base_input = base_model.input
# base_output = base_model.layers[-2].output

# final_output = layers.Dense(128)(base_output)
# final_output = layers.Activation('relu')(final_output)
# final_output = layers.Dense(64)(final_output)
# final_output = layers.Activation('relu')(final_output)
# final_output = layers.Dense(7, activation='softmax')(final_output)

# model = keras.Model(inputs=base_input, outputs=final_output)

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# model.summary()
