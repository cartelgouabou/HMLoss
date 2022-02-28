#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Properly implemented ResNet-32 model for CIFAR-10 as described in paper [1].

Arthur Foahom. November 2021.
"""
import tensorflow as tf
from tensorflow.keras.layers import Conv2D,BatchNormalization,Activation,add,Input,GlobalAveragePooling2D,Dense
from tensorflow.keras.models import Model
import numpy as np




# ResNet building block of two layers
def building_block(X, filter_size, filters,weight_decay,kernel_initializer, stride=1):

    # Save the input value for shortcut
    X_shortcut = X

    # Reshape shortcut for later adding if dimensions change
    if stride > 1:

        X_shortcut = Conv2D(filters, (1, 1), strides=stride,kernel_regularizer=tf.keras.regularizers.l2(weight_decay), padding='same', kernel_initializer=kernel_initializer)(X_shortcut)
        X_shortcut = BatchNormalization(axis=3)(X_shortcut)

    # First layer of the block
    X = Conv2D(filters, kernel_size = filter_size, strides=stride, padding='same', kernel_initializer=kernel_initializer,kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(X)
    X = BatchNormalization(axis=3,fused=True)(X)
    X = Activation('relu')(X)

    # Second layer of the block
    X = Conv2D(filters, kernel_size = filter_size, strides=(1, 1),kernel_regularizer=tf.keras.regularizers.l2(weight_decay), padding='same', kernel_initializer=kernel_initializer)(X)
    X = BatchNormalization(axis=3)(X)
    X = add([X, X_shortcut])  # Add shortcut value to main path
    X = Activation('relu')(X)

    return X


# Full model
def ResNet32(input_shape, num_classes, name,loss_type,weight_decay,kernel_initializer):

    # Define the input
    X_input = Input(input_shape)

    # Stage 1
    X = Conv2D(filters=16, kernel_size=3, strides=(1, 1),kernel_regularizer=tf.keras.regularizers.l2(2*0.0001),padding='same', kernel_initializer='he_normal')(X_input)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    # Stage 2
    X = building_block(X, filter_size=3, filters=16,weight_decay=weight_decay,kernel_initializer=kernel_initializer, stride=1)
    X = building_block(X, filter_size=3, filters=16,weight_decay=weight_decay,kernel_initializer=kernel_initializer, stride=1)
    X = building_block(X, filter_size=3, filters=16,weight_decay=weight_decay,kernel_initializer=kernel_initializer, stride=1)
    X = building_block(X, filter_size=3, filters=16,weight_decay=weight_decay,kernel_initializer=kernel_initializer, stride=1)
    X = building_block(X, filter_size=3, filters=16,weight_decay=weight_decay,kernel_initializer=kernel_initializer, stride=1)

    # Stage 3
    X = building_block(X, filter_size=3, filters=32,weight_decay=weight_decay,kernel_initializer=kernel_initializer, stride=2)  # dimensions change (stride=2)
    X = building_block(X, filter_size=3, filters=32,weight_decay=weight_decay,kernel_initializer=kernel_initializer, stride=1)
    X = building_block(X, filter_size=3, filters=32,weight_decay=weight_decay,kernel_initializer=kernel_initializer, stride=1)
    X = building_block(X, filter_size=3, filters=32,weight_decay=weight_decay,kernel_initializer=kernel_initializer, stride=1)
    X = building_block(X, filter_size=3, filters=32,weight_decay=weight_decay,kernel_initializer=kernel_initializer, stride=1)

    # Stage 4
    X = building_block(X, filter_size=3, filters=64,weight_decay=weight_decay,kernel_initializer=kernel_initializer, stride=2)  # dimensions change (stride=2)
    X = building_block(X, filter_size=3, filters=64,weight_decay=weight_decay,kernel_initializer=kernel_initializer, stride=1)
    X = building_block(X, filter_size=3, filters=64,weight_decay=weight_decay,kernel_initializer=kernel_initializer, stride=1)
    X = building_block(X, filter_size=3, filters=64,weight_decay=weight_decay,kernel_initializer=kernel_initializer, stride=1)
    X = building_block(X, filter_size=3, filters=64,weight_decay=weight_decay,kernel_initializer=kernel_initializer, stride=1)

    # Average pooling and output layer
    X = GlobalAveragePooling2D()(X)
    
    if loss_type=='focal':
        pi=1/num_classes
        X = Dense(num_classes,activation='softmax',bias_initializer=tf.keras.initializers.Constant(value=-np.log(1-pi)/pi))(X)
    elif loss_type=='logits':
        X = Dense(num_classes)(X)
    else:
        X = Dense(num_classes,activation='softmax')(X)
        

    # Create model
    model = Model(inputs=X_input, outputs=X, name=name)

    return model

