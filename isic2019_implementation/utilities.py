#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 17:36:52 2021

@author: arthur
"""

import numpy as np

from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from glob import glob
def score_gen(img_paths,model,target_size):
    from tensorflow.keras.preprocessing.image import load_img,img_to_array
    score_predict=[]
    for i in range(len(img_paths)):
        img=load_img(img_paths[i], target_size=target_size,interpolation='bicubic')
        x=img_to_array(img) 
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        out=model.predict(x)
        if i==0:
            score_predict=out
        else :
            score_predict=np.concatenate((score_predict,out))
    return score_predict


def path_list(test_path,structured,image_size=(100,100)):
    if structured==True:  #data are structured by class
        datanorm = ImageDataGenerator(preprocessing_function = preprocess_input)
        data = datanorm.flow_from_directory(test_path, target_size=image_size,interpolation='bicubic', shuffle=False, batch_size=1)
        test_filenames = data.filenames
        test_pathnames=data.filepaths
        y_test=data.labels
    else:
         test_pathnames = glob(test_path +'*.JPG')
         if (len(test_pathnames)==0):
             test_pathnames = glob(test_path +'*.jpg')
         test_filenames=[]
         cursor=len(test_path)
         for i in range(len(test_pathnames)):
             test_filenames.append(test_pathnames[i][cursor:-4])
         y_test=[]
    return test_pathnames,test_filenames,y_test
         
def read_and_preprocess(data_path,image_size, preprocess_input):
   datanorm = ImageDataGenerator(preprocessing_function = preprocess_input)
    # we need to see the data in the same order
    # for both predictions and targets
   data = datanorm.flow_from_directory(data_path, target_size=image_size,interpolation='bicubic', shuffle=False, batch_size=1)
   num=300
   numBen=len(data.classes[data.classes==0])
   numMal=len(data.classes[data.classes==1])
   if numMal>num:
       ben_idx = np.random.randint(low=0, high=numBen-1, size=300)
       mal_idx = np.random.randint(low=numBen, high=numMal+numBen, size=300)
   else:
       ben_idx = np.random.randint(low=0, high=numBen-1, size=300)
       mal_idx = np.random.randint(low=numBen, high=numMal+numBen, size=numMal)
   idx=np.concatenate((ben_idx, mal_idx),axis=0)
   data_filenames = data.filenames
   X = []
   target = []
   i=0
   for idx_ in idx:
       x=[]
       y=[]
       x=data[idx_][0]
       y=data[idx_][1]
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
   return X,target,data_filenames
  