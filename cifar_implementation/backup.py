# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 10:35:17 2021

@author: arthu
"""


from __future__ import print_function, division
import tensorflow as tf
from numpy.random import seed
seed(12)

import numpy as np
import pandas as pd

from tensorflow.keras.callbacks import Callback
from time import time
#from keras.utils.np_utils import to_categorical





class LinearWarmUpScheduler(Callback):
  def __init__(self,initLr,checkpoint_root,filenames,epochs):
    super(LinearWarmUpScheduler, self).__init__()
    #self.lr_schedule = schedule
    self.count = 0
    self.delay = [5,160,180]
    self.lr_multiplier=[1,0.1,0.01]
    self.linear_warm_up=0
    # Initialize the best as infinity.
    self.best = np.Inf
    self.initLr=initLr
    self.lr_list=[self.initLr*x for x in self.lr_multiplier]
    self.filenames=filenames
    self.path_weights=checkpoint_root+'best_weights/'
    self.path_history=checkpoint_root+'training_history/'
    self.ep=epochs
    self.training_time =0
    self.start_time=0
      
  def on_epoch_begin(self, epoch, logs=None):
      if (epoch==0) :
            self.start_time=time()
      self.linear_warm_up=(self.initLr*epoch/self.delay[0])
      if (epoch < self.delay[0]):
          self.linear_warm_up=(self.initLr*epoch/self.delay[0])
          tf.keras.backend.set_value(self.model.optimizer.lr, self.linear_warm_up)
          print('Learning increase on Epoch %d: new Learning rate is %s.' % (epoch,str(round(self.linear_warm_up,8))))
      else:
          if (epoch==self.delay[self.count]) :
              tf.keras.backend.set_value(self.model.optimizer.lr, self.lr_list[self.count])
              print('Learning rate decreased on Epoch %d: new Learning rate is %s.' % (epoch,str(round(self.lr_list[self.count],8))))
              self.count+=1
              if self.count>=(len(self.delay)-1):
                  self.count=len(self.delay)-1
  def on_epoch_end(self, epoch, logs=None):
      if (epoch == (self.ep-1)):
               print('\model training stopped at epoch{}'.format(epoch)) 
               self.model.save_weights(self.path_weights+'best_weights_'+self.filenames+'.hdf5')
               self.best_training_time = time()-self.start_time
               self.wait=0
               callbacks_save=pd.DataFrame(np.array([[self.best_training_time]]),columns=['training_time'])
               callbacks_save.to_csv(self.path_history+'best_callbacks_'+self.filenames+'.csv')
      return 