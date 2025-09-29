# import
import sys
from matplotlib.colors import cnames
import pandas as pd
import numpy as np
import os
from sqlalchemy import null
from tqdm import tqdm
import sklearn as sk
import tensorflow as tf
from pathlib import Path
import cv2
import pydicom
from PIL import Image
# import tensorflow.keras as keras
from tensorflow import keras

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, Model


def Create_MLP(in_Dim, outDim = 2, metrics = ['accuracy'], loss = "binary_crossentropy", optimizer = "adam", regression = True):
    
    #model definition
    mlp = Sequential()
    mlp.add(Dense(32,input_dim=in_Dim,activation='relu'))
    # model.add(Dropout(0.7))
    mlp.add(Dense(16,activation='relu'))
    # model.add(Dropout(0.8))
    mlp.add(Dense(8,activation='relu'))
    # model.add(Dropout(0.8))
    mlp.add(Dense(4,activation='relu'))
    # model.add(Dropout(0.5))
    #can continue the model

    if regression:
        mlp.add(Dense(outDim,activation='sigmoid'))

    
    mlp.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
    )
    
    return mlp


def fit_mlp(X_tr,Y_tr,X_val,Y_val,model, checkpointPath, ep = 100, bSize = 32, personalizeCheckpoint = False, checkpointList = []):
    # CREATE CALLBACKS

    callbacks_list = checkpointList
    if not personalizeCheckpoint:
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
    						checkpointPath / Path('mlp'), 
    						monitor='val_accuracy', verbose=1, 
    						save_best_only=True, mode='max')
        callbacks_list = [checkpoint]


    history = model.fit(
            x = X_tr,
            y = Y_tr,
            epochs = ep,
            validation_data = [X_val,Y_val],
            callbacks = callbacks_list,    #some problems with DS layers and callbacks
            batch_size=bSize,
    )

    return history


def load_best_mlp(checkpointPaht, loadDefault = True):
    model = null
    if loadDefault:
        model = tf.keras.models.load_model(checkpointPaht / Path('mlp'))
    else:
        model = tf.keras.models.load_model(checkpointPaht)
    return model
    