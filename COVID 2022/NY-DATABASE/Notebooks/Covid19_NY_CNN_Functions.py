
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
import tensorflow_addons as tfa

#IMPORT ALL LAYERS AND KERAS/TENSORFLOW PARAMS
from tensorflow.keras import layers
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPool2D, Dense, Flatten, Input, Concatenate, Add, AveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2




sys.path.append('../E-CNN-classifier-main/libs')
import ds_layer #Dempster-Shafer layer
import utility_layer_train #Utility layer for training
import utility_layer_test #Utility layer for training
import AU_imprecision #Metric average utility for set-valued classification


#     return lout
def DSmassCalcLayer(lin, prototypes, num_class):
    # print(lin.shape[1])
    ED = ds_layer.DS1(prototypes,lin.shape[1])(lin)
    ED_ac = ds_layer.DS1_activate(prototypes)(ED)
    mass_prototypes = ds_layer.DS2(prototypes, num_class)(ED_ac)
    mass_omega = ds_layer.DS2_omega(prototypes, num_class)(mass_prototypes)
    # mass_Dempster = ds_layer.DS3_Dempster(prototypes, num_class)(mass_omega)
    lout = ds_layer.DS3_Dempster(prototypes, num_class)(mass_omega)
    # lout = ds_layer.DS3_normalize()(mass_Dempster)

    return lout


#     return lout
def MassCalcLayer_ASI(lin, num_class):

    weights_x = ds_layer.L1_wadd(lin.shape[1], num_class)(lin)
    distances = ds_layer.L1_wadd_activate()(weights_x)
    distances = BatchNormalization()(distances)
    # massesCalc = ds_layer.L2_masses()(distances)

    return distances




def ResFBlock(x,filt,kernel,stride):
    x = Conv2D(filters=filt, kernel_size=(kernel,kernel),strides=(stride,stride), kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(0.0005))(x)
    x = BatchNormalization()(x)
    x = layers.ReLU()(x)
    # x = Dropout(0.6)(x)
    return x



def Create_CNN(inputShape = (224,224,3), lastActivation = 'sigmoid', outDim = 2, metrics = ['accuracy'], loss = "binary_crossentropy", optimizer = "adam"):
    
    
    inputs = Input(shape=inputShape)

    # x = Rescaling(1.0 / 255)(inputs)

    x = ResFBlock(inputs,16,5,2)
    
    block1 = ResFBlock(x,32,3,2)
    block1 = ResFBlock(block1,32,3,1)
    # block1 = Dropout(0.7)(block1)
    b1_pass = ResFBlock(x,32,1,2)
    # b1_pass = Dropout(0.7)(b1_pass)
    block1 = Add()([block1,b1_pass])
    block1 = AveragePooling2D(pool_size=(2, 2), strides=(2,2), padding='same')(block1)

    block2 = ResFBlock(block1,48,3,1)
    block2 = ResFBlock(block2,48,3,1)
    # block2 = Dropout(0.7)(block2)
    b2_pass = ResFBlock(block1,48,1,1)
    # b2_pass = Dropout(0.7)(b2_pass)
    block2 = Add()([block2,b2_pass])
    block2 = AveragePooling2D(pool_size=(2, 2), strides=(2,2), padding='same')(block2)

    block3 = ResFBlock(block2,64,3,1)
    block3 = ResFBlock(block3,64,3,1)
    # block3 = Dropout(0.7)(block3)
    b3_pass = ResFBlock(block2,64,1,1)
    # b3_pass = Dropout(0.7)(b3_pass)
    block3 = Add()([block3,b3_pass])
    block3 = AveragePooling2D(pool_size=(2, 2), strides=(2,2), padding='same')(block3)

    block4 = ResFBlock(block3,80,3,1)
    block4 = ResFBlock(block4,80,3,1)
    # block4 = Dropout(0.7)(block4)
    b4_pass = ResFBlock(block3,80,1,1)
    # b4_pass = Dropout(0.7)(b4_pass)
    block4 = Add()([block4,b4_pass])
    block4 = AveragePooling2D(pool_size=(2, 2), strides=(2,2), padding='same')(block4)

    block5 = ResFBlock(block4,96,3,1)
    block5 = ResFBlock(block5,96,3,1)
    # block5 = Dropout(0.7)(block5)
    b5_pass = ResFBlock(block4,96,1,1)
    # b5_pass = Dropout(0.7)(b5_pass)
    block5 = Add()([block5,b5_pass])
    block5 = AveragePooling2D(pool_size=(2, 2), strides=(2,2), padding='same')(block5)


    block1 = MaxPool2D(pool_size=(4,4), strides=(4,4))(block1)
    block2 = MaxPool2D(pool_size=(2,2), strides=(2,2))(block2)
    
    block1 = Flatten()(block1)
    block2 = Flatten()(block2)
    block3 = Flatten()(block3)
    block4 = Flatten()(block4)
    block5 = Flatten()(block5)
    
    massFusion = Concatenate()([block1,block2,block3,block4,block5])

    print(massFusion)

    outputs = Dense(32, activation='relu')(massFusion)
    outputs = Dense(8, activation='relu')(outputs)


    outputs = Dense(outDim, activation=lastActivation)(outputs)

    cnn =  Model(inputs=inputs, outputs=outputs)


    cnn.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
    )
    
    return cnn

def Create_CNN_ASI(inputShape = (224,224,3), outDim = 2, prototypes = 40, metrics = ['accuracy'], loss = "binary_crossentropy", optimizer = "adam"):
    
    inputs = Input(shape=inputShape)

    x = ResFBlock(inputs,16,5,2)
    block1 = ResFBlock(x,32,3,2)
    block1 = ResFBlock(block1,32,3,1)
    # block1 = Dropout(0.6)(block1)
    b1_pass = ResFBlock(x,32,1,2)
    # b1_pass = Dropout(0.6)(b1_pass)
    block1 = Add()([block1,b1_pass])
    block1 = AveragePooling2D(pool_size=(2, 2), strides=(2,2), padding='same')(block1)

    block2 = ResFBlock(block1,48,3,1)
    block2 = ResFBlock(block2,48,3,1)
    # block2 = Dropout(0.7)(block2)
    b2_pass = ResFBlock(block1,48,1,1)
    # b2_pass = Dropout(0.7)(b2_pass)
    block2 = Add()([block2,b2_pass])
    block2 = AveragePooling2D(pool_size=(2, 2), strides=(2,2), padding='same')(block2)

    block3 = ResFBlock(block2,64,3,1)
    block3 = ResFBlock(block3,64,3,1)
    # block3 = Dropout(0.7)(block3)
    b3_pass = ResFBlock(block2,64,1,1)
    # b3_pass = Dropout(0.7)(b3_pass)
    block3 = Add()([block3,b3_pass])
    block3 = AveragePooling2D(pool_size=(2, 2), strides=(2,2), padding='same')(block3)

    block4 = ResFBlock(block3,80,3,1)
    block4 = ResFBlock(block4,80,3,1)
    # block4 = Dropout(0.7)(block4)
    b4_pass = ResFBlock(block3,80,1,1)
    # b4_pass = Dropout(0.7)(b4_pass)
    block4 = Add()([block4,b4_pass])
    block4 = AveragePooling2D(pool_size=(2, 2), strides=(2,2), padding='same')(block4)

    block5 = ResFBlock(block4,96,3,1)
    block5 = ResFBlock(block5,96,3,1)
    # block5 = Dropout(0.7)(block5)
    b5_pass = ResFBlock(block4,96,1,1)
    # b5_pass = Dropout(0.7)(b5_pass)
    block5 = Add()([block5,b5_pass])
    block5 = AveragePooling2D(pool_size=(2, 2), strides=(2,2), padding='same')(block5)


    block1 = MaxPool2D(pool_size=(4,4), strides=(4,4))(block1)
    block2 = MaxPool2D(pool_size=(2,2), strides=(2,2))(block2)
    # block3 = MaxPool2D(pool_size=(2,2), strides=(2,2))(block3)
    
    block1 = Flatten()(block1)
    block1 = MassCalcLayer_ASI(block1, num_class=outDim)
    block2 = Flatten()(block2)
    block2 = MassCalcLayer_ASI(block2, num_class=outDim)
    block3 = Flatten()(block3)
    block3 = MassCalcLayer_ASI(block3, num_class=outDim)
    block4 = Flatten()(block4)
    block4 = MassCalcLayer_ASI(block4, num_class=outDim)
    block5 = Flatten()(block5)
    block5 = MassCalcLayer_ASI(block5, num_class=outDim)
    

    massFusion = Concatenate(axis=1)([block1,block2,block3,block4,block5])

    # CALCOLO DELLE MASSE A PARTIRE DALLA LISTA DI MASSE DI OGNI LIVELLO
    print(massFusion.shape)
    weights_x = ds_layer.L1_wadd(massFusion.shape[1], outDim)(massFusion)
    print(weights_x.shape)
    distances = ds_layer.L1_wadd_activate()(weights_x)
    # print(distances.shape)
    massesCalc = ds_layer.L2_masses()(distances)
    # massesCalc = ds_layer.L2_masses(outDim)(weights_x)
    print(massesCalc.shape)
    finalMasses = ds_layer.L3_combine_masses()(massesCalc)
    print(finalMasses.shape)
    
    # #Utility layer for testing
    outputs = utility_layer_train.DM_pignistic(outDim)(finalMasses)

    cnn = Model(inputs=inputs, outputs=outputs)
    

    cnn.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
    )


    return cnn

def Create_CNN_with_DS(inputShape = (224,224,3), outDim = 2, prototypes = 40, metrics = ['accuracy'], loss = "binary_crossentropy", optimizer = "adam"):
    
    inputs = Input(shape=inputShape)

    x = ResFBlock(inputs,16,5,2)
    block1 = ResFBlock(x,32,3,2)
    block1 = ResFBlock(block1,32,3,1)
    # block1 = Dropout(0.6)(block1)
    b1_pass = ResFBlock(x,32,1,2)
    # b1_pass = Dropout(0.6)(b1_pass)
    block1 = Add()([block1,b1_pass])
    block1 = AveragePooling2D(pool_size=(2, 2), strides=(2,2), padding='same')(block1)

    block2 = ResFBlock(block1,48,3,1)
    block2 = ResFBlock(block2,48,3,1)
    # block2 = Dropout(0.7)(block2)
    b2_pass = ResFBlock(block1,48,1,1)
    # b2_pass = Dropout(0.7)(b2_pass)
    block2 = Add()([block2,b2_pass])
    block2 = AveragePooling2D(pool_size=(2, 2), strides=(2,2), padding='same')(block2)

    block3 = ResFBlock(block2,64,3,1)
    block3 = ResFBlock(block3,64,3,1)
    # block3 = Dropout(0.7)(block3)
    b3_pass = ResFBlock(block2,64,1,1)
    # b3_pass = Dropout(0.7)(b3_pass)
    block3 = Add()([block3,b3_pass])
    block3 = AveragePooling2D(pool_size=(2, 2), strides=(2,2), padding='same')(block3)

    block4 = ResFBlock(block3,80,3,1)
    block4 = ResFBlock(block4,80,3,1)
    # block4 = Dropout(0.7)(block4)
    b4_pass = ResFBlock(block3,80,1,1)
    # b4_pass = Dropout(0.7)(b4_pass)
    block4 = Add()([block4,b4_pass])
    block4 = AveragePooling2D(pool_size=(2, 2), strides=(2,2), padding='same')(block4)

    block5 = ResFBlock(block4,96,3,1)
    block5 = ResFBlock(block5,96,3,1)
    # block5 = Dropout(0.7)(block5)
    b5_pass = ResFBlock(block4,96,1,1)
    # b5_pass = Dropout(0.7)(b5_pass)
    block5 = Add()([block5,b5_pass])
    block5 = AveragePooling2D(pool_size=(2, 2), strides=(2,2), padding='same')(block5)


    block1 = MaxPool2D(pool_size=(4,4), strides=(4,4))(block1)
    block2 = MaxPool2D(pool_size=(2,2), strides=(2,2))(block2)
    # block3 = MaxPool2D(pool_size=(2,2), strides=(2,2))(block3)
    
    block1 = Flatten()(block1)
    block1 = DSmassCalcLayer(block1, prototypes=prototypes, num_class=outDim)
    block2 = Flatten()(block2)
    block2 = DSmassCalcLayer(block2, prototypes=prototypes, num_class=outDim)
    block3 = Flatten()(block3)
    block3 = DSmassCalcLayer(block3, prototypes=prototypes, num_class=outDim)
    block4 = Flatten()(block4)
    block4 = DSmassCalcLayer(block4, prototypes=prototypes, num_class=outDim)
    block5 = Flatten()(block5)
    block5 = DSmassCalcLayer(block5, prototypes=prototypes, num_class=outDim)
    

    massFusion = Concatenate()([block1,block2,block3,block4,block5])

    # DA DS2 IN DS3
    newNprot = (outDim+1)*5
    mass_prototypes = ds_layer.DS2(newNprot, outDim)(massFusion)
    mass_omega = ds_layer.DS2_omega(newNprot, outDim)(mass_prototypes)
    mass_Dempster = ds_layer.DS3_Dempster(newNprot, outDim)(mass_omega)
    mass_Dempster_normalize = ds_layer.DS3_normalize()(mass_Dempster)
    
    # #Utility layer for testing
    outputs = utility_layer_train.DM_pignistic(outDim)(mass_Dempster_normalize)

    cnn = Model(inputs=inputs, outputs=outputs)
    

    cnn.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
    )


    return cnn

def Create_CNN_with_Lrog(inputShape = (224,224,3), outDim = 2, prototypes = 40, metrics = ['accuracy'], loss = "binary_crossentropy", optimizer = "adam"):
    
    inputs = Input(shape=inputShape)

    x = ResFBlock(inputs,16,5,2)
    block1 = ResFBlock(x,32,3,2)
    block1 = ResFBlock(block1,32,3,1)
    # block1 = Dropout(0.6)(block1)
    b1_pass = ResFBlock(x,32,1,2)
    # b1_pass = Dropout(0.6)(b1_pass)
    block1 = Add()([block1,b1_pass])
    block1 = AveragePooling2D(pool_size=(2, 2), strides=(2,2), padding='same')(block1)

    block2 = ResFBlock(block1,48,3,1)
    block2 = ResFBlock(block2,48,3,1)
    # block2 = Dropout(0.7)(block2)
    b2_pass = ResFBlock(block1,48,1,1)
    # b2_pass = Dropout(0.7)(b2_pass)
    block2 = Add()([block2,b2_pass])
    block2 = AveragePooling2D(pool_size=(2, 2), strides=(2,2), padding='same')(block2)

    block3 = ResFBlock(block2,64,3,1)
    block3 = ResFBlock(block3,64,3,1)
    # block3 = Dropout(0.7)(block3)
    b3_pass = ResFBlock(block2,64,1,1)
    # b3_pass = Dropout(0.7)(b3_pass)
    block3 = Add()([block3,b3_pass])
    block3 = AveragePooling2D(pool_size=(2, 2), strides=(2,2), padding='same')(block3)

    block4 = ResFBlock(block3,80,3,1)
    block4 = ResFBlock(block4,80,3,1)
    # block4 = Dropout(0.7)(block4)
    b4_pass = ResFBlock(block3,80,1,1)
    # b4_pass = Dropout(0.7)(b4_pass)
    block4 = Add()([block4,b4_pass])
    block4 = AveragePooling2D(pool_size=(2, 2), strides=(2,2), padding='same')(block4)

    block5 = ResFBlock(block4,96,3,1)
    block5 = ResFBlock(block5,96,3,1)
    # block5 = Dropout(0.7)(block5)
    b5_pass = ResFBlock(block4,96,1,1)
    # b5_pass = Dropout(0.7)(b5_pass)
    block5 = Add()([block5,b5_pass])
    block5 = AveragePooling2D(pool_size=(2, 2), strides=(2,2), padding='same')(block5)


    block1 = MaxPool2D(pool_size=(4,4), strides=(4,4))(block1)
    block2 = MaxPool2D(pool_size=(2,2), strides=(2,2))(block2)
    # block3 = MaxPool2D(pool_size=(2,2), strides=(2,2))(block3)

    # block1 = BatchNormalization()(block1)
    # block2 = BatchNormalization()(block2)
    # block3 = BatchNormalization()(block3)
    # block4 = BatchNormalization()(block4)
    # block5 = BatchNormalization()(block5)
    
    block1 = Flatten()(block1)
    # block1 = DSmassCalcLayer(block1, prototypes=prototypes, num_class=outDim)
    block2 = Flatten()(block2)
    # block2 = DSmassCalcLayer(block2, prototypes=prototypes, num_class=outDim)
    block3 = Flatten()(block3)
    # block3 = DSmassCalcLayer(block3, prototypes=prototypes, num_class=outDim)
    block4 = Flatten()(block4)
    # block4 = DSmassCalcLayer(block4, prototypes=prototypes, num_class=outDim)
    block5 = Flatten()(block5)
    # block5 = DSmassCalcLayer(block5, prototypes=prototypes, num_class=outDim)
    

    massFusion = Concatenate()([block1,block2,block3,block4,block5])


    # DA DS2 IN DS3
    print(massFusion.shape)
    weights_x = ds_layer.L1_wadd(massFusion.shape[1], outDim)(massFusion)
    print(weights_x.shape)
    distances = ds_layer.L1_wadd_activate(massFusion.shape[1], outDim)(weights_x)
    # print(distances.shape)
    massesCalc = ds_layer.L2_masses(outDim)(distances)
    # massesCalc = ds_layer.L2_masses(outDim)(weights_x)
    print(massesCalc.shape)
    finalMasses = ds_layer.L3_combine_masses(outDim)(massesCalc)
    print(finalMasses.shape)
    
    # #Utility layer for testing
    outputs = utility_layer_train.DM_pignistic(outDim)(finalMasses)

    cnn = Model(inputs=inputs, outputs=outputs)
    

    cnn.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
    )


    return cnn





def fit_cnn(X_tr,Y_tr,X_val,Y_val,model, checkpointPath, ep = 100, bSize = 32, personalizeCheckpoint = False, checkpointList = [], weights = {0: 0.5, 1: 0.5}, datagen = null):
    # CREATE CALLBACKS

    callbacks_list = checkpointList
    if not personalizeCheckpoint:
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
	    					checkpointPath / Path('cnn'), 
	    					monitor='val_accuracy', 
	    					save_best_only=True, mode='max')
        callbacks_list = [checkpoint]

    if datagen != null:
        history = model.fit(
            datagen.flow(
                X_tr,
                Y_tr,
                batch_size=bSize, 
                seed=42,
                shuffle=True),
            epochs = ep,
            validation_data = [X_val,Y_val],
            callbacks = callbacks_list,
            batch_size=bSize,
            class_weight= weights
        )
    else:
        history = model.fit(
            x = X_tr,
            y = Y_tr,
            epochs = ep,
            validation_data = [X_val,Y_val],
            callbacks = callbacks_list,
            batch_size=bSize,
            class_weight= weights
        )
    return history


def load_best_cnn(checkpointPaht, loadDefault = True, customObjects = null):
    model = null
    
    if customObjects != null:
        if loadDefault:
            model = tf.keras.models.load_model(checkpointPaht / Path('cnn'), custom_objects=customObjects)
        else:
            model = tf.keras.models.load_model(checkpointPaht, custom_objects=customObjects)
    else:
        if loadDefault:
            model = tf.keras.models.load_model(checkpointPaht / Path('cnn'))
        else:
            model = tf.keras.models.load_model(checkpointPaht)
    
    return model
    