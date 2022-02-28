#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 3 15:06:47 2021

Imbalanced CIFAR 

@author: Arthur Cartel Foahom Gouabou
Created imbalanced dataset with specific imbalanced ratio and imbalanced type.
Use this class either for cifar10 and cifar100.
  The class requires the following inputs:
  - `dataset_name` (name of the dataset to modified): cifar10 or cifar100 
  - `imb_type` (type of imbalance): Specified the type of imbalance to do, between long tailed and step
  - `imb_factor` (ratio of imbalance): denote the raio between samples sizes of the most frequent and least frequent class
  

    References:
        my_paper
                                              
    Standalone usage:
        from imbalance_cifar import IMBALANCECIFAR
        
        imbalanceData = IMBALANCECIFAR(dataset_name='cifar100',  imb_type='step', imb_factor=0.01)
        x_train=imbalanceData.train_images
        y_train=imbalanceData.train_labels
        x_test=imbalanceData.test_images
        y_test=imbalanceData.test_labels
        num_per_cls=imbalanceData.num_per_cls_dict 


"""
import numpy as np
from tensorflow.keras import datasets

class IMBALANCECIFAR:
    
   
    def __init__(self,dataset_name='cifar10',  imb_type='exp', imb_factor=0.01):
        if    (dataset_name!='cifar10') & (dataset_name!='cifar100'):
            raise Exception("specified the right dataset name between 'cifar10' or 'cifar100', you wrote: {}".format(dataset_name))
        if dataset_name=='cifar10':
            (self.train_images, self.train_labels), (self.test_images, self.test_labels) = datasets.cifar10.load_data()
        else:
            (self.train_images, self.train_labels), (self.test_images, self.test_labels) = datasets.cifar100.load_data()
        self.test_labels=np.array(self.test_labels,dtype=np.int64)
        self.test_labels=np.reshape(self.test_labels,-1)
        img_num_list = self.get_img_num_per_cls(imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)
        
        
    def get_img_num_per_cls(self, imb_type, imb_factor):    
        self.cls_num=len(np.unique(self.train_labels))
        img_max= len(self.train_images) / self.cls_num
        img_num_per_cls=[]
        if imb_type == 'exp':
            for cls_idx in range(self.cls_num):
                num = img_max * (imb_factor**(cls_idx / (self.cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(self.cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(self.cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * self.cls_num)
        return img_num_per_cls
    
    def gen_imbalanced_data(self,img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.train_labels, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.train_images[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.train_images = new_data
        self.train_labels = np.array(new_targets)
        #return self.train_images,self.train_labels, self.test_images, self.test_labels

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list



