import os
import zipfile
import random
import shutil
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
from os import getcwd
from os import listdir
import cv2
from tensorflow.keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image  as mpimg
from pathlib import Path
import keras

curr_dir=str(Path.cwd())
main_dir=curr_dir[0:curr_dir.rfind("/")]



def make_network():
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(128, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
       ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    return model



def save_model(model):
    os.makedirs(main_dir+'/saved_models/')
    model.save(main_dir+'/saved_models/mask_detector.h5')




def create_generators():

    train_data_directory=main_dir+'/datasets/augmented/'
    validation_data_directory=main_dir+'/datasets/augmented/'

    model=make_network()

    train_datagen = ImageDataGenerator(rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2)

    train_generator = train_datagen.flow_from_directory(
    train_data_directory,
    target_size=(150, 150),
    batch_size=10,
    class_mode='binary',
    subset='training')

    validation_generator = train_datagen.flow_from_directory(
    validation_data_directory, # same directory as training data
    target_size=(150, 150),
    batch_size=10,
    class_mode='binary',
    subset='validation')


    model.fit_generator(
     train_generator,
     steps_per_epoch = train_generator.samples // 10,
     validation_data = validation_generator,
     validation_steps = validation_generator.samples // 10,
     epochs = 20)

    save_model(model)






create_generators()
