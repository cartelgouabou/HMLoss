#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Properly implemented pretrained EfficientNetB3 and ..... as described in paper [1].
Reference:
[1] my paper
If you use this implementation in you work, please don't forget to mention the
author, Arthur Cartel Foahom Gouabou.
'''

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB3
import numpy as np
def EfficientNet(ratio_freeze,path_weights,load):
    model_O=EfficientNetB3(input_shape=(300, 300, 3),weights='imagenet',include_top=True) #'weights=imagenet'
    num_layer=len(model_O.layers)
    num_freeze_layer= (ratio_freeze*num_layer)//100
    model_O.layers.pop()
    prediction=model_O.layers[-2].output
    prediction = Dense(2, kernel_initializer='he_normal')(prediction)
    model = Model(inputs=model_O.input, outputs=prediction)
    for layer in model.layers:
             layer.trainable = True
    for layer in model.layers[:-num_freeze_layer]:
            layer.trainable = False  
    if load==True:
            model.load_weights(path_weights)
    return model

def EfficientNetBias(ratio_freeze,path_weights,load,last_layer_activation='sigmoid'):
    model_O=EfficientNetB3(input_shape=(300, 300, 3),weights='imagenet',include_top=True) #'weights=imagenet'
    num_layer=len(model_O.layers)
    pi=0.01
    num_freeze_layer= (ratio_freeze*num_layer)//100
    model_O.layers.pop()
    prediction=model_O.layers[-2].output
    prediction = Dense(1, kernel_initializer='he_normal',activation=last_layer_activation,bias_initializer=tf.keras.initializers.Constant(value=-np.log((1-pi)/pi)))(prediction)
    model = Model(inputs=model_O.input, outputs=prediction)
    for layer in model.layers:
             layer.trainable = True
    for layer in model.layers[:-num_freeze_layer]:
            layer.trainable = False  
    if load==True:
            model.load_weights(path_weights)
    return model