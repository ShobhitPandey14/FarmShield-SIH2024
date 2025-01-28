import os
import numpy as np
import tensorflow as tf
from keras_preprocessing import image
from keras import applications
from keras import layers
from keras import Model

# Directories for dataset
train_dir = "E:\General\Programming\Python\ML from Scratch\SIHModel\Datasets\dataset3\dataset3\Train\Train"
val_dir = "E:\General\Programming\Python\ML from Scratch\SIHModel\Datasets\dataset3\dataset3\Validation\Validation"

# Debugging: Print the contents of the directories
print("Training data classes found:", os.listdir(train_dir))
print("Validation data classes found:", os.listdir(val_dir))

# Image data generator for augmentation
train_datagen = image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

val_datagen = image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Print class indices to check if classes are loaded
print("Classes found in training data:", train_generator.class_indices)

# Define and compile the model
base_model = applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = layers.Dense(256, activation='relu')(x)
x = layers.GlobalAveragePooling2D()(x)
predictions = layers.Dense(train_generator.num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
for layer in base_model.layers:
    layer.trainable = True
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_generator, validation_data=val_generator, epochs=5)

from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

import seaborn as sns
sns.set_theme()
sns.set_context("poster")

figure(figsize=(25, 25), dpi=100)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

model.save("E:\General\Programming\Python\ML from Scratch\SIHModel\model.h5")
model.save_weights("E:\General\Programming\Python\ML from Scratch\SIHModel\model.weights.h5")