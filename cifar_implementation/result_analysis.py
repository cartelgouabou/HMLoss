#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 16:59:18 2021

@author: arthur
"""

import argparse
import os
import pandas as pd
import numpy as np
from resnet_cifar import ResNet32
from tensorflow.keras.utils import to_categorical
from imbalance_cifar import IMBALANCECIFAR
from utilities import read_and_preprocess
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
root_path=os.getcwd()
parser = argparse.ArgumentParser(description='Tensorflow Cifar Generate results')
parser.add_argument('--dataset','--dataset_name', default='cifar10', help='dataset setting',dest='dataset_name')
parser.add_argument('--loss_function_list', nargs='+', type=str, help='list of loss functions to evaluate',dest='case_list')
parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
parser.add_argument('--imb_ratio_list','--ir_list', nargs='+', type=float, help='list of imbalance ratio to evaluate',dest='ir_list')
parser.add_argument('--num_runs', default=3, type=int, help='number of runs to launch ')
parser.add_argument('--wd', '--weight-decay', default=0.0002, type=float,
                    metavar='W', help='weight decay fro convolutions (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--kernel-initializer', '--k-init', default='he_normal', type=str,help='define kernel initializer (default: he_normal)',
                    dest='kernel_initializer')
args = parser.parse_args()

print('List of case to evaluate:')
print(args.case_list)
print('List of imbalance ratio to evaluate:')
print(args.ir_list)
path_resultat =root_path+'/RESULTATS/'
if not os.path.exists(path_resultat):
      os.makedirs(path_resultat)
imbalanceData = IMBALANCECIFAR(dataset_name=args.dataset_name,  imb_type=args.imb_type, imb_factor=1)
x_train=imbalanceData.train_images
num_class=imbalanceData.cls_num
input_shape = list(x_train.shape[1:])
x_test=imbalanceData.test_images
y_test=imbalanceData.test_labels
y_test_2D= to_categorical(y_test, num_classes=num_class)
run_list=[]
for i in range(args.num_runs):
    run_list.append('run'+str(i))
metric_list=['ACC','errRate','AUC','BACC','Training_time']

[x_test,y_test]=read_and_preprocess(len(x_test), x_test, y_test_2D)


for case in args.case_list:
    for ir in args.ir_list:
        path_checkpoint=root_path+'/'+case+'/checkpoint'
        data=[]
        for run in run_list:
            filenames=args.dataset_name+'_'+args.imb_type+'_'+case+'_ir_'+str(ir)+'_'+run
            path_weights_load= path_checkpoint+'/best_weights/'+'best_weights_'+filenames+'.hdf5'
            callback=pd.read_csv(path_checkpoint+'/training_history/'+'best_callbacks_'+filenames+'.csv')
            training_time=callback.training_time[0]
            loss_type='softmax'
            if (case=='flg05')|(case=='flg1')|(case=='flg2')|(case=='flg3')|(case=='flg05a25')|(case=='flg1a25')|(case=='flg2a25')|(case=='flg05a50')|(case=='flg1a50')|(case=='flg2a50')|(case=='flg05a75')|(case=='flg1a75')|(case=='flg2a75'):
                loss_type='focal'
            elif (case=='LDAM')|(case=='CBLDAM')|(case=='csLDAM'):
                loss_type='logits'
            model=ResNet32(input_shape=input_shape , num_classes=num_class, name='ResNet-32',loss_type=loss_type,weight_decay=args.weight_decay,kernel_initializer=args.kernel_initializer)
            model.load_weights(path_weights_load)
            y_score=model.predict(x_test)
            y_pred=[np.argmax(y_score[i]) for i in np.arange(len(y_score))]
            data.append([accuracy_score(y_test, y_pred),1-accuracy_score(y_test, y_pred),roc_auc_score(y_test_2D, y_score),balanced_accuracy_score(y_test, y_pred),\
                            training_time])    
        data=np.array(data)
        data=np.transpose(data)
        cv=pd.DataFrame(data,
                        index=metric_list,
                        columns=run_list)
        name_result='statistic_'+args.dataset_name+'_'+args.imb_type+'_'+case+'_ir_'+str(ir)+'.csv'
        cv.to_csv(path_resultat+name_result)
    
