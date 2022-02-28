#!/usr/bin/env python
# coding: utf-8
"""
Created on Tue Oct 19 15:06:47 2021

@author: Arthur Cartel Foahom Gouabou

Implements of loss functions used in the paper [Adressing class imbalance with Hard Mining Loss](https://) 

Binary class setting implementation

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
from tensorflow.keras import backend as K
import numpy as np


from tensorflow.keras.losses import Loss

#perspective: dynamic weighting per batch


def HingeLoss():
    return tf.keras.losses.Hinge()
  #tf.keras.losses.Reduction.NONE
class FocalLoss(Loss):
    def __init__(self,gamma=2,alpha=None,name='focal_loss'):
        super().__init__()
        self.gamma=gamma
        self.alpha=alpha
        if alpha==None:
             self.alpha=1.
        else:
             self.alpha=alpha

    
    def call(self,y_true,y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        # Force prediction value to range between epsilon and 1-epsilon to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        
        logits=K.log(tf.math.divide(y_pred,(1-y_pred))) #sigmoid inverse
        
        neg_abs_logits = -tf.abs(logits)
        relu_logits = tf.where(logits > 0, logits, 0)

        cross_entropy = relu_logits - logits*y_true + K.log(1 + K.exp(neg_abs_logits))
        #cross_entropy=K.mean(loss_vec)
        
        pt = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        factor=K.pow((1 - pt), self.gamma)
        pos_weights=tf.where(K.equal(y_true, 1), self.alpha, 1 - self.alpha)
        focal_loss=factor*cross_entropy
        return pos_weights*focal_loss
   

class BinaryCrossentropy(Loss):
    def __init__(self,class_weight=None,name='binary_cross_entropy_loss'):
        super().__init__()
        if class_weight==None:
            self.pos_weight=1.
            self.neg_weight=1.
        else:
            self.pos_weight=class_weight[1]
            self.neg_weight=class_weight[0]

    
    def call(self,y_true,y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        # Force prediction value to range between epsilon and 1-epsilon to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        
        logits=K.log(tf.math.divide(y_pred,(1-y_pred))) #sigmoid inverse
        
        neg_abs_logits = -tf.abs(logits)
        relu_logits = tf.where(logits > 0, logits, 0)
        cross_entropy = relu_logits - logits*y_true + K.log(1 + K.exp(neg_abs_logits))
        weights_factor=tf.where(K.equal(y_true, 1), self.pos_weight, self.neg_weight)
        return weights_factor*cross_entropy

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
        #  labels are already one hot encoded
        index_float = y_true
        batch_m = tf.matmul(self.m_list[None, :], tf.transpose(index_float))
        batch_m = tf.reshape(batch_m, (-1, 1))
        logits_m = logits - batch_m

        # if condition is true, return x_m[index], otherwise return x[index]
        index_bool = tf.cast(index_float, tf.bool)
        output = tf.where(index_bool, logits_m, logits)

        #print("labels : \n", labels, "\n logits : \n", logits)
        # Calculate Cross Entropy with news logits
        #epsilon = K.epsilon()
        #y_pred_m=tf.nn.softmax(output*self.s)
        #y_pred_m=y_pred_m/K.sum(y_pred_m,axis=-1,keepdims=True)
        #y_pred_m = K.clip(y_pred_m, epsilon, 1. - epsilon)
        #loss = -tf.multiply(y_true * K.log(y_pred_m),self.class_weight)
        # Clip the prediction value to prevent NaN's and Inf's
        loss=tf.nn.softmax_cross_entropy_with_logits(y_true,output*self.s)

        return tf.reduce_mean(loss)#K.mean(K.sum(loss, axis=-1))

 
class BinaryHardMiningLoss(Loss):
    """Computs the hardmining loss between the labels and predictions.
    
    Implements the hard mining loss function (HMLoss) for a binary setting with sigmoid.
    
    This is the official implementation of the hard mining loss function (HMLoss) that was first introduced in the paper
    (https://arxiv.org/pdf/1708.02002.pdf). HMLoss is extremely useful for classification when you have highly imbalanced classes.
    Use this implementation of HMLoss when there are two ore more label classes. It expect labels to be provided in a 'one_hot' representation.
    
    Standalone usage:
    
    >>> y_true = [1,1,0]
    >>> y_pred = [0.8,0.9,0.9]
    >>> bhml = BinaryHardMiningLoss(alpha=0.75,delta=100000,gamma=1)
    >>> bhml(y_true,y_pred).numpy()
    0.20466684
    
    Usage with the 'compile()' API:
        
     ```python   
     model.compile(optimizer='sgd',loss=BinaryHardMiningLoss(alpha=.75, gamma=1, detlta=100000)) 
     ```
    """
    
    
 
    def __init__(self,delta=100000,gamma=1,alpha=None,name='Hard_mining_loss'):
        super().__init__()
        self.gamma=gamma
        self.delta=delta
        self.alpha=alpha
        if alpha==None:
             self.alpha=1.
        else:
             self.alpha=alpha
        """Initializes `BinaryHardMiningLoss` instance.
        Args:
            alpha: balancing factor, default value is None.
            delta: thresholding of outlier, default value is 1e5
            gamma: modulating factor, default value is 1.0.
        """


    def call(self,y_true,y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        # Force prediction value to range between epsilon and 1-epsilon to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        logits=K.log(tf.math.divide(y_pred,(1-y_pred))) #sigmoid inverse

        neg_abs_logits = -tf.abs(logits)
        relu_logits = tf.where(logits > 0, logits, 0)

        cross_entropy = relu_logits - logits*y_true + K.log(1 + K.exp(neg_abs_logits))

        pt = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        factor=K.pow(((tf.sin(pt*np.pi)/(pt*np.pi))-(tf.exp(-self.delta*pt)*(tf.sin(pt*np.pi*self.delta)/(pt*np.pi*self.delta)))), self.gamma)
        weights_factor=tf.where(K.equal(y_true, 1), self.alpha, 1 - self.alpha)
        hm_loss = factor*cross_entropy
        return weights_factor*hm_loss

