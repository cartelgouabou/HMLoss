# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 17:41:19 2020

@author: arthu
"""
#ration_freeze :percentage of layer to finetune
#w type of weight initialisation 1 for imagenet and 0 for random
#
import numpy as np
from losses import FocalLoss,BinaryCrossentropy,HingeLoss,LDAMLoss,BinaryHardMiningLoss

def model_compiler(model,case_loss_fn,img_num_per_class):
    if (case_loss_fn=='bce'): 
        model.compile(loss=BinaryCrossentropy(),
                      optimizer='adam',
                      metrics=['accuracy'])
    elif (case_loss_fn=='hinge')|(case_loss_fn=='cshinge'):
        model.compile(loss=HingeLoss(),
                      optimizer='adam',
                      metrics=['accuracy'])
    elif (case_loss_fn=='LDAM')|(case_loss_fn=='LDAM-RW'): 
        model.compile(loss=LDAMLoss(cls_num_list=img_num_per_class),
                      optimizer='adam',
                      metrics=['accuracy'])
    elif (case_loss_fn=='csbce')|(case_loss_fn=='CBLoss'): 
        beta=0.9999
        class_weight=[]
        if (case_loss_fn=='csbce'):
            for i,j in zip(np.arange(2),img_num_per_class):
                class_weight.append(1/j)
        if (case_loss_fn=='CBLoss'):
            for i,j in zip(np.arange(2),img_num_per_class):
                class_weight.append((1-beta)/(1-beta**j))
        model.compile(loss=BinaryCrossentropy(class_weight=class_weight),
                      optimizer='adam',
                      metrics=['accuracy'])
    elif (case_loss_fn=='flg05')|(case_loss_fn=='flg1')|(case_loss_fn=='flg2')|(case_loss_fn=='CBFLossg05')|(case_loss_fn=='CBFLossg1')|(case_loss_fn=='CBFLossg2'):
        if (case_loss_fn=='flg05')|(case_loss_fn=='CBFLossg05'):
            gamma=0.5
        elif (case_loss_fn=='flg1')|(case_loss_fn=='CBFLossg1'):
            gamma=1
        elif (case_loss_fn=='flg2')|(case_loss_fn=='CBFLossg2'):
            gamma=2
        model.compile(loss=FocalLoss(gamma=gamma),
                      optimizer='adam',
                      metrics=['accuracy'])
    elif (case_loss_fn=='flg05a25')|(case_loss_fn=='flg1a25')|(case_loss_fn=='flg2a25')|(case_loss_fn=='flg05a50')|(case_loss_fn=='flg1a50')|(case_loss_fn=='flg2a50')|(case_loss_fn=='flg05a75')|(case_loss_fn=='flg1a75')|(case_loss_fn=='flg2a75'):
        if (case_loss_fn=='flg05a25')|(case_loss_fn=='flg05a50')|(case_loss_fn=='flg05a75'):
            gamma=0.5
            if (case_loss_fn=='flg05a25'):
                alpha=0.25
            elif (case_loss_fn=='flg05a50'):
                alpha=0.5
            elif (case_loss_fn=='flg05a75'):
                alpha=0.75
        elif (case_loss_fn=='flg1a25')|(case_loss_fn=='flg1a50')|(case_loss_fn=='flg1a75'):
            gamma=1
            if (case_loss_fn=='flg1a25'):
                alpha=0.25
            elif (case_loss_fn=='flg1a50'):
                alpha=0.5
            elif (case_loss_fn=='flg1a75'):
                alpha=0.75
        elif (case_loss_fn=='flg2a25')|(case_loss_fn=='flg2a50')|(case_loss_fn=='flg2a75'):
            gamma=2
            if (case_loss_fn=='flg2a25'):
                alpha=0.25
            elif (case_loss_fn=='flg2a50'):
                alpha=0.5
            elif (case_loss_fn=='flg2a75'):
                alpha=0.75
        model.compile(loss=FocalLoss(gamma=gamma,alpha=alpha),
                      optimizer='adam',
                      metrics=['accuracy'])
    elif (case_loss_fn=='hmld10000g05a25')|(case_loss_fn=='hmld10000g1a25')|(case_loss_fn=='hmld10000g2a25')|(case_loss_fn=='hmld10000g05a50')|(case_loss_fn=='hmld10000g1a50')|(case_loss_fn=='hmld10000g2a50')|(case_loss_fn=='hmld10000g05a75')|(case_loss_fn=='hmld10000g1a75')|(case_loss_fn=='hmld10000g2a75'):
        delta=10000
        if (case_loss_fn=='hmld10000g05a25')|(case_loss_fn=='hmld10000g05a50')|(case_loss_fn=='hmld10000g05a75'):
            gamma=0.5
            if (case_loss_fn=='hmld10000g05a25'):
                alpha=0.25
            elif (case_loss_fn=='hmld10000g05a50'):
                alpha=0.5
            elif (case_loss_fn=='hmld10000g05a75'):
                alpha=0.75
        elif (case_loss_fn=='hmld10000g1a25')|(case_loss_fn=='hmld10000g1a50')|(case_loss_fn=='hmld10000g1a75'):
            gamma=1
            if (case_loss_fn=='hmld10000g1a25'):
                alpha=0.25
            elif (case_loss_fn=='hmld10000g1a50'):
                alpha=0.5
            elif (case_loss_fn=='hmld10000g1a75'):
                alpha=0.75
        elif (case_loss_fn=='hmld10000g2a25')|(case_loss_fn=='hmld10000g2a50')|(case_loss_fn=='hmld10000g2a75'):
            gamma=2
            if (case_loss_fn=='hmld10000g2a25'):
                alpha=0.25
            elif (case_loss_fn=='hmld10000g2a50'):
                alpha=0.5
            elif (case_loss_fn=='hmld10000g2a75'):
                alpha=0.75
        model.compile(loss=BinaryHardMiningLoss(gamma=gamma,alpha=alpha,delta=delta),
                      optimizer='adam',
                      metrics=['accuracy'])

    elif (case_loss_fn=='hmld100000g05a25')|(case_loss_fn=='hmld100000g1a25')|(case_loss_fn=='hmld100000g2a25')|(case_loss_fn=='hmld100000g05a50')|(case_loss_fn=='hmld100000g1a50')|(case_loss_fn=='hmld100000g2a50')|(case_loss_fn=='hmld100000g05a75')|(case_loss_fn=='hmld100000g1a75')|(case_loss_fn=='hmld100000g2a75'):
        delta=100000
        if (case_loss_fn=='hmld100000g05a25')|(case_loss_fn=='hmld100000g05a50')|(case_loss_fn=='hmld100000g05a75'):
            gamma=0.5
            if (case_loss_fn=='hmld100000g05a25'):
                alpha=0.25
            elif (case_loss_fn=='hmld100000g05a50'):
                alpha=0.5
            elif (case_loss_fn=='hmld100000g05a75'):
                alpha=0.75
        elif (case_loss_fn=='hmld100000g1a25')|(case_loss_fn=='hmld100000g1a50')|(case_loss_fn=='hmld100000g1a75'):
            gamma=1
            if (case_loss_fn=='hmld100000g1a25'):
                alpha=0.25
            elif (case_loss_fn=='hmld100000g1a50'):
                alpha=0.5
            elif (case_loss_fn=='hmld100000g1a75'):
                alpha=0.75
        elif (case_loss_fn=='hmld100000g2a25')|(case_loss_fn=='hmld100000g2a50')|(case_loss_fn=='hmld100000g2a75'):
            gamma=2
            if (case_loss_fn=='hmld100000g2a25'):
                alpha=0.25
            elif (case_loss_fn=='hmld100000g2a50'):
                alpha=0.5
            elif (case_loss_fn=='hmld100000g2a75'):
                alpha=0.75
        model.compile(loss=BinaryHardMiningLoss(gamma=gamma,alpha=alpha,delta=delta),
                      optimizer='adam',
                      metrics=['accuracy'])

    elif (case_loss_fn=='hmld100g05')|(case_loss_fn=='hmld1000g05')|(case_loss_fn=='hmld10000g05')|(case_loss_fn=='hmld100000g05')|(case_loss_fn=='hmld1000000g05')|(case_loss_fn=='hmld10000000g05')|(case_loss_fn=='hmld100000000g05')|(case_loss_fn=='hmld1000000000g05')|(case_loss_fn=='hmld10000000000g05')|(case_loss_fn=='hmld100000000000g05')|(case_loss_fn=='hmld1000000000000g05'):
        gamma=0.5
        if (case_loss_fn=='hmld100g05'):
            delta=100
        elif (case_loss_fn=='hmld1000g05'):
            delta=1000
        elif (case_loss_fn=='hmld10000g05'):
            delta=10000
        elif (case_loss_fn=='hmld100000g05'):
            delta=100000
        elif (case_loss_fn=='hmld1000000g05'):
            delta=1000000
        elif (case_loss_fn=='hmld10000000g05'):
            delta=10000000    
        elif (case_loss_fn=='hmld100000000g05'):
            delta=100000000
        elif (case_loss_fn=='hmld1000000000g05'):
            delta=1000000000
        elif (case_loss_fn=='hmld10000000000g05'):
            delta=10000000000
        elif (case_loss_fn=='hmld100000000000g05'):
            delta=100000000000
        elif (case_loss_fn=='hmld1000000000000g05'):
            delta=1000000000000
        model.compile(loss=BinaryHardMiningLoss(delta=delta,gamma=gamma),
                      optimizer='adam',
                      metrics=['accuracy']) 

    elif (case_loss_fn=='hmld100g1')|(case_loss_fn=='hmld1000g1')|(case_loss_fn=='hmld10000g1')|(case_loss_fn=='hmld100000g1')|(case_loss_fn=='hmld1000000g1')|(case_loss_fn=='hmld10000000g1')|(case_loss_fn=='hmld100000000g1')|(case_loss_fn=='hmld1000000000g1')|(case_loss_fn=='hmld10000000000g1')|(case_loss_fn=='hmld100000000000g1')|(case_loss_fn=='hmld1000000000000g1'):
        gamma=1
        if (case_loss_fn=='hmld100g1'):
            delta=100
        elif (case_loss_fn=='hmld1000g1'):
            delta=1000
        elif (case_loss_fn=='hmld10000g1'):
            delta=10000
        elif (case_loss_fn=='hmld100000g1'):
            delta=100000
        elif (case_loss_fn=='hmld1000000g1'):
            delta=1000000
        elif (case_loss_fn=='hmld10000000g1'):
            delta=10000000    
        elif (case_loss_fn=='hmld100000000g1'):
            delta=100000000
        elif (case_loss_fn=='hmld1000000000g1'):
            delta=1000000000
        elif (case_loss_fn=='hmld10000000000g1'):
            delta=10000000000
        elif (case_loss_fn=='hmld100000000000g1'):
            delta=100000000000
        elif (case_loss_fn=='hmld1000000000000g1'):
            delta=1000000000000
        model.compile(loss=BinaryHardMiningLoss(delta=delta,gamma=gamma),
                      optimizer='adam',
                      metrics=['accuracy']) 

    elif (case_loss_fn=='hmld100g2')|(case_loss_fn=='hmld1000g2')|(case_loss_fn=='hmld10000g2')|(case_loss_fn=='hmld100000g2')|(case_loss_fn=='hmld1000000g2')|(case_loss_fn=='hmld10000000g2')|(case_loss_fn=='hmld100000000g2')|(case_loss_fn=='hmld1000000000g2')|(case_loss_fn=='hmld10000000000g2')|(case_loss_fn=='hmld100000000000g2')|(case_loss_fn=='hmld1000000000000g2'):
        gamma=2
        if (case_loss_fn=='hmld100g2'):
            delta=100
        elif (case_loss_fn=='hmld1000g2'):
            delta=1000
        elif (case_loss_fn=='hmld10000g2'):
            delta=10000
        elif (case_loss_fn=='hmld100000g2'):
            delta=100000
        elif (case_loss_fn=='hmld1000000g2'):
            delta=1000000
        elif (case_loss_fn=='hmld10000000g2'):
            delta=10000000    
        elif (case_loss_fn=='hmld100000000g2'):
            delta=100000000
        elif (case_loss_fn=='hmld1000000000g2'):
            delta=1000000000
        elif (case_loss_fn=='hmld10000000000g2'):
            delta=10000000000
        elif (case_loss_fn=='hmld100000000000g2'):
            delta=100000000000
        elif (case_loss_fn=='hmld1000000000000g2'):
            delta=1000000000000
        model.compile(loss=BinaryHardMiningLoss(delta=delta,gamma=gamma),
                      optimizer='adam',
                      metrics=['accuracy']) 
    else:
        print('Error: specified the right loss function name, check list of function name specification on function_name_list.txt')
       
            

def train_model(model,case_loss_fn,train_gen,valid_gen,callback1,callback2,train_size,valid_size,batch_size,epochs,img_num_per_class):
    if (case_loss_fn=='LDAM-RW')|(case_loss_fn=='cshinge')|(case_loss_fn=='CBFLossg05')|(case_loss_fn=='CBFLossg1')|(case_loss_fn=='CBFLossg2'):
        class_weight=dict()
        beta=0.9999
        if (case_loss_fn=='LDAM-RW')|(case_loss_fn=='cshinge'):
            for i,j in zip(np.arange(2),img_num_per_class):
                class_weight[i]=1/j
        if (case_loss_fn=='CBFLossg05')|(case_loss_fn=='CBFLossg1')|(case_loss_fn=='CBFLossg2') :
            for i,j in zip(np.arange(2),img_num_per_class):
                class_weight[i]=(1-beta)/(1-beta**j)
        resultat=model.fit(train_gen,
                             steps_per_epoch=train_size//batch_size,
                             epochs=epochs ,
                             verbose=0,
                             validation_data= valid_gen,
                             validation_steps=valid_size//batch_size,
                             class_weight=class_weight,
                             callbacks=[callback1,callback2]
                             )
    else:
        resultat=model.fit(train_gen,
                             steps_per_epoch=train_size//batch_size,
                             epochs=epochs ,
                             verbose=0,
                             validation_data= valid_gen,
                             validation_steps=valid_size//batch_size,
                             callbacks=[callback1,callback2]
                             )
    return resultat
    

