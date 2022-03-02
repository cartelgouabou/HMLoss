#!/usr/bin/env python
# coding: utf-8
"""
Created on Tue Oct 19 15:06:47 2021

@author: Arthur Cartel Foahom Gouabou

Implements of loss functions used in the paper [Adressing class imbalance with Hard Mining Loss](https://) 

Multiclass setting implementation

Computes HMLoss between true labels and predicted labels.
Use this loss either for two or more label classes applications.
  The loss function requires the following inputs:
  - `y_true` (true label): 
  - `y_pred` (predicted value): This is the model's prediction, i.e, a single  floating-point value which either represents a
    [logit](https://en.wikipedia.org/wiki/Logit), (i.e, value in [-inf, inf]
    when `from_logits=True`) or a probability (i.e, value in [0., 1.] when
    `from_logits=False`).

    References:
        my_paper
                                              
    Standalone usage:
        model.compile(loss=[robust_loss(thresold=.25, gamma=2, pos_weight=)], metrics=["accuracy"], optimizer=adam)  


"""

import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.losses import Loss


def CategoricalCrossentropy(class_weight=None):
    #class_weight=class_weight
    #from_logits=from_logits
    if class_weight!=None:
              class_weight=tf.constant(class_weight,dtype=tf.float32)
    else:
        class_weight=tf.constant(1,dtype=tf.float32)
              

    def categorical_cross_entropy(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """
        epsilon = K.epsilon()
        # Clip the prediction value to prevent NaN's and Inf's
        y_pred=y_pred/K.sum(y_pred,axis=-1,keepdims=True)
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        # Calculate Cross Entropy
        cross_entropy = -tf.multiply(y_true * K.log(y_pred),class_weight)

        # Calculate Focal Loss
        #loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Compute mean loss in mini_batch
        return K.mean(K.sum(cross_entropy, axis=-1))

    return categorical_cross_entropy



def CategoricalFocalLoss(alpha=None, gamma=2.):
    if alpha==None:
              alpha=tf.constant(1,dtype=tf.float32)
    else:
              alpha=tf.constant(alpha,dtype=tf.float32)
    

    def categorical_focal_loss(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred=y_pred/K.sum(y_pred,axis=-1,keepdims=True)
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -tf.multiply(y_true * K.log(y_pred),alpha)

        # Calculate Focal Loss
        loss = K.pow(1 - y_pred, gamma) * cross_entropy

        # Compute mean loss in mini_batch
        return K.mean(K.sum(loss, axis=-1))

    return categorical_focal_loss


    
class LDAMLoss(Loss):

    def __init__(self, cls_num_list, max_m=0.5, class_weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = tf.convert_to_tensor(m_list, dtype=tf.float32)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.n_classes = len(cls_num_list)
        if class_weight==None:
              self.class_weight=tf.constant(1,dtype=tf.float32)
        else:
              self.class_weight=tf.constant(class_weight,dtype=tf.float32)


    def call(self, y_true, logits):
        #  labels are one hot encoded
        index_float = y_true
        batch_m = tf.matmul(self.m_list[None, :], tf.transpose(index_float))
        batch_m = tf.reshape(batch_m, (-1, 1))
        logits_m = logits - batch_m

        # if condition is true, return logits_m[index], otherwise return logits[index]
        index_bool = tf.cast(index_float, tf.bool)
        output = tf.where(index_bool, logits_m, logits)

        # Calculate Cross Entropy with news logits
        loss=tf.nn.softmax_cross_entropy_with_logits(y_true,output*self.s)

        return tf.reduce_mean(loss)
    

class CategoricalHardMiningLoss(Loss):
    """Computs the hardmining loss between the labels and predictions.
    
    This is the official implementation of the hard mining loss function (HMLoss) that was first introduced in the paper
    (https://arxiv.org/pdf/1708.02002.pdf). HMLoss is extremely useful for classification when you have highly imbalanced classes.
    Use this implementation of HMLoss when there are two ore more label classes. It expect labels to be provided in a 'one_hot' representation.
    
    Standalone usage:
    
    >>> y_true = [[1,0,0], [1,0,0], [0,1,0],[0,0,1]]
    >>> y_pred = [[0.8,0.1,0.1], [0.9,0.1,0.0], [0.1,0.1,0.8],[0.1,0.2,0.7]]
    >>> hml = CategoricalHardMiningLoss(alpha=0.75,delta=10000000,gamma=1)
    >>> hml(y_true,y_pred).numpy()
    0.46121484
    
    Usage with the 'compile()' API:
        
     ```python   
     model.compile(optimizer='sgd',loss=CategoricalHardMiningLoss(alpha=.75, gamma=1, detlta=10000000)) 
     ```
    """
    """
    Args:
      alpha: balancing factor, default value is None.
      delta: thresholding of outlier, default value is 1e7
      gamma: modulating factor, default value is 1.0.
    """
    
 
    def __init__(self,delta=10000000,alpha=None, gamma=1.,name='Hard_mining_loss'):
        super().__init__()
        self.gamma=gamma
        self.delta=delta
        self.alpha=alpha
        if alpha==None:
              self.alpha=tf.constant(1,dtype=tf.float32)
        else:
              self.alpha=tf.constant(alpha,dtype=tf.float32)


    def call(self,y_true,y_pred):
        """"
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """
        # Force prediction value to range between epsilon and 1-epsilon to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred=y_pred/K.sum(y_pred,axis=-1,keepdims=True)
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -tf.multiply(y_true * K.log(y_pred),self.alpha)

        # Calculate hard mining loss
        loss=K.pow( (tf.sin((y_pred)*np.pi)/((y_pred)*np.pi))-( (tf.exp(-self.delta*y_pred))*(tf.sin((y_pred)*self.delta*np.pi)/((y_pred)*self.delta*np.pi)))  , self.gamma) * cross_entropy
        
        # Compute mean loss in mini_batch
        return K.mean(K.sum(loss, axis=-1))
