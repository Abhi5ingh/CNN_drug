import os
import sys
import tensorflow as tf
import logging
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras import layers, models, optimizers
from src.utils import ensure_dir
from src.components.model import build_model

def train_cnn(dataset_dir="dataset",
              img_size=(224,224),
              batch_size=32,epochs=5,
              model_save_path="cnn_tki_classifier.h5"):
    """Trains a CNN model on images in dataset_dir and saves the model to model_save_path."""
    logging.info(f"Training CNN model on images in {dataset_dir}")
    if not os.path.exists(dataset_dir):
        logging.error(f"Dataset directory {dataset_dir} not found")
        return False
    
    train_datagen=ImageDataGenerator(rescale=1./255,
                                     validation_split=0.2,
                                     rotation_range=20,
                                     zoom_range=0.2,
                                     width_shift_range=0.1,
                                     height_shift_range=0.1,
                                     horizontal_flip=True,
                                     fill_mode='nearest')
    
    train_generator=train_datagen.flow_from_directory(dataset_dir,
                                                      target_size=img_size,
                                                      batch_size=batch_size,
                                                      class_mode='binary',
                                                      subset='training',
                                                      shuffle=True)
    

    val_generator=train_datagen.flow_from_directory(dataset_dir,
                                                      target_size=img_size,
                                                      batch_size=batch_size,
                                                      class_mode='binary',
                                                      subset='validation',
                                                      shuffle=False)
    model = build_model(img_size,channels=3)
    model.compile(optimizers.Adam(lr=0.0001),loss='binary_crossentropy',metrics=['accuracy'])

    model.summary(print_fn=logging.info)

    history=model.fit(train_generator,
                      steps_per_epoch=train_generator.samples//batch_size,
                      validation_data=val_generator,
                      validation_steps=val_generator.samples//batch_size,
                      epochs=epochs)
    model.save(model_save_path)
    logging.info(f"Model saved to {model_save_path}")

    return model
