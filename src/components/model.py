import logging
import os
import sys
from keras import layers, models, optimizers

def build_model(input_shape=(224,224),channels=3):
    """Build a CNN model for image classification."""
    img_height,img_width = input_shape

    model = models.Sequential([
        layers.Conv2D(32,(3,3),activation='relu',input_shape=(img_height,img_width,channels)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64,(3,3),activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(128,(3,3),activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128,activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1,activation='sigmoid')
    ])
    logging.info("Built CNN model")
    return model