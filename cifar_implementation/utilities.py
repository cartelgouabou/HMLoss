#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 17:36:52 2021

@author: arthur
"""

import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator

         
def read_and_preprocess(dataset_size, x_test, y_test):
   datanorm = ImageDataGenerator(rescale=1./255)
    # we need to see the data in the same order
    # for both predictions and targets
   data = datanorm.flow(x_test, y_test,batch_size=1)
   X = []
   target = []
   i=0
   for x, y in data: 
       i+=1
       s1=x
       #t1=y
       t1=np.argmax(y, axis=1)
       if i==1:
          X=x
          #target=t1
          target=t1
       else :
          X=np.concatenate((X,s1))
          target=np.concatenate((target,t1))
          if i==dataset_size:
              break
   return X,target

