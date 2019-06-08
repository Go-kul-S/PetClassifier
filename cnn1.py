# -*- coding: utf-8 -*-
"""
Created on Sat May 18 18:34:21 2019

@author: Goki
"""

#this is a convolution neural network to classify the given images as cats or dogs

#Importing the packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#initializing
classifier = Sequential()

#the convolution process
#If using a theano backend, the iput_shape = (3, 64, 64)
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu' ))
#MaxPooling layer
classifier.add(MaxPooling2D(pool_size = (2,2)))

#adding a second convolution layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu' ))
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Flattening
classifier.add(Flatten())
#Adding the fully connected ANN two hidden layers
classifier.add(Dense(activation="relu", units=128))
classifier.add(Dense(activation="relu", units=128))
classifier.add(Dense(activation="sigmoid", units=1))


#compiling the NN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ["accuracy"])

#fitting the image inputs
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_dataset = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_dataset = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64 , 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        train_dataset,
        samples_per_epoch=8000,
        epochs=25,
        validation_data=test_dataset,
        nb_val_samples = 2000)