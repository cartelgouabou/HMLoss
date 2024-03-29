#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 19:56:20 2022

@author: arthur
"""

import torch
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import os
from sklearn.metrics import balanced_accuracy_score,confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root_path=os.getcwd()

# this function is used during training process, to calculation the loss and accuracy
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def compute_img_mean_std(image_paths):
    """
        computing the mean and std of three channel on the whole dataset,
        first we should normalize the image from 0-255 to 0-1
    """

    img_h, img_w = 300,300
    imgs = []
    means, stdevs = [], []

    for i in tqdm(range(len(image_paths))):
        img = cv2.imread(image_paths[i])
        img = cv2.resize(img, (img_h, img_w))
        imgs.append(img)

    imgs = np.stack(imgs, axis=3)
    print(imgs.shape)

    imgs = imgs.astype(np.float32) / 255.

    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()  # resize to one row
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    means.reverse()  # BGR --> RGB
    stdevs.reverse()

    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))
    return means,stdevs

def prepare_folders(args):
    
    folders_util = [root_path+args.history_path, root_path+args.model_path,root_path+args.result_dir,
                    os.path.join(root_path+args.history_path, args.store_name),
                    os.path.join(root_path+args.result_dir, args.store_name),
                    os.path.join(root_path+args.model_path, args.store_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)
            


def bal_accuracy(val_loader, model, args):
    
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            _, pred = torch.max(output, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    bacc = balanced_accuracy_score(all_targets, all_preds).astype(float)
    return bacc
    
def calc_confusion_mat(val_loader, model, args):
    
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            _, pred = torch.max(output, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    cf = confusion_matrix(all_targets, all_preds).astype(float)

    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)

    cls_acc = cls_hit / cls_cnt

    print('Class Accuracy : ')
    print(cls_acc)
    classes = [str(x) for x in args.cls_num_list]
    plot_confusion_matrix(all_targets, all_preds, classes)
    plt.savefig(os.path.join(args.path_history, args.store_name, 'confusion_matrix.png'))

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def eval_distribution(y_val,y_pred_val,epoch):
    nmel=np.sum(y_val)
    nnev=len(y_val)-nmel
    mel_pos=np.array(y_val==1)
    nev_pos=np.array(y_val==0)
    y_pred_val=y_pred_val[:,1]
    mel_pred=y_pred_val*mel_pos
    mel_pred=np.delete(mel_pred, np.where(mel_pred==0.))
    nev_pred=y_pred_val*nev_pos
    nev_pred=np.delete(nev_pred, np.where(nev_pred==0.))
    y_pred_val_2=np.concatenate((1-nev_pred,mel_pred),axis=None)
    mel_00_005=np.sum((mel_pred>=0.0) & (mel_pred<0.05))
    mel_005_01=np.sum((mel_pred>=0.05) & (mel_pred<0.1))
    mel_01_02=np.sum((mel_pred>=0.1) & (mel_pred<0.2))
    mel_02_03=np.sum((mel_pred>=0.2) & (mel_pred<0.3))
    mel_03_04=np.sum((mel_pred>=0.3) & (mel_pred<0.4))
    mel_04_05=np.sum((mel_pred>=0.4) & (mel_pred<0.5))
    mel_05_06=np.sum((mel_pred>=0.5) & (mel_pred<0.6))
    mel_06_07=np.sum((mel_pred>=0.6) & (mel_pred<0.7))
    mel_07_08=np.sum((mel_pred>=0.7) & (mel_pred<0.8))
    mel_08_09=np.sum((mel_pred>=0.8) & (mel_pred<0.9))
    mel_09_1=np.sum((mel_pred>=0.9) & (mel_pred<1.))
    nev_00_005=np.sum((nev_pred>=0.0) & (nev_pred<0.05))
    nev_005_01=np.sum((nev_pred>=0.05) & (nev_pred<0.1))
    nev_01_02=np.sum((nev_pred>=0.1) & (nev_pred<0.2))
    nev_02_03=np.sum((nev_pred>=0.2) & (nev_pred<0.3))
    nev_03_04=np.sum((nev_pred>=0.3) & (nev_pred<0.4))
    nev_04_05=np.sum((nev_pred>=0.4) & (nev_pred<0.5))
    nev_05_06=np.sum((nev_pred>=0.5) & (nev_pred<0.6))
    nev_06_07=np.sum((nev_pred>=0.6) & (nev_pred<0.7))
    nev_07_08=np.sum((nev_pred>=0.7) & (nev_pred<0.8))
    nev_08_09=np.sum((nev_pred>=0.8) & (nev_pred<0.9))
    nev_09_1=np.sum((nev_pred>=0.9) & (nev_pred<1.))
    dist_00_005 = np.sum((y_pred_val_2>=0.0) & (y_pred_val_2<0.05))
    dist_005_01 = np.sum((y_pred_val_2>=0.05) & (y_pred_val_2<0.1))
    dist_01_02  = np.sum((y_pred_val_2>=0.1) & (y_pred_val_2<0.2))
    dist_02_03  = np.sum((y_pred_val_2>=0.2) & (y_pred_val_2<0.3))
    dist_03_04  = np.sum((y_pred_val_2>=0.3) & (y_pred_val_2<0.4))
    dist_04_05  = np.sum((y_pred_val_2>=0.4) & (y_pred_val_2<0.5))
    dist_05_06  = np.sum((y_pred_val_2>=0.5) & (y_pred_val_2<0.6))
    dist_06_07  = np.sum((y_pred_val_2>=0.6) & (y_pred_val_2<0.7))
    dist_07_08  = np.sum((y_pred_val_2>=0.7) & (y_pred_val_2<0.8))
    dist_08_09  = np.sum((y_pred_val_2>=0.8) & (y_pred_val_2<0.9))
    dist_09_1   = np.sum((y_pred_val_2>=0.9) & (y_pred_val_2<1.))

    mel_new_row=pd.DataFrame(np.array([[epoch,mel_00_005,mel_005_01,mel_01_02,mel_02_03,mel_03_04,mel_04_05,mel_05_06,mel_06_07,mel_07_08,mel_08_09,mel_09_1,nmel]]),
                                                 columns=['Epoch','0.0-0.05','0.05-0.1','0.1-0.2','0.2-0.3','0.3-0.4','0.4-0.5','0.5-0.6','0.6-0.7','0.7-0.8','0.8-0.9','0.9-1','Total mel'])
    nev_new_row=pd.DataFrame(np.array([[epoch,nev_00_005,nev_005_01,nev_01_02,nev_02_03,nev_03_04,nev_04_05,nev_05_06,nev_06_07,nev_07_08,nev_08_09,nev_09_1,nnev]]),
                                                 columns=['Epoch','0.0-0.05','0.05-0.1','0.1-0.2','0.2-0.3','0.3-0.4','0.4-0.5','0.5-0.6','0.6-0.7','0.7-0.8','0.8-0.9','0.9-1','Total nev'])
    proba_new_row=pd.DataFrame(np.array([[epoch,dist_00_005,dist_005_01,dist_01_02,dist_02_03,dist_03_04,dist_04_05,dist_05_06,dist_06_07,dist_07_08,dist_08_09,dist_09_1,nnev+nmel]]),
                                                 columns=['Epoch','0.0-0.05','0.05-0.1','0.1-0.2','0.2-0.3','0.3-0.4','0.4-0.5','0.5-0.6','0.6-0.7','0.7-0.8','0.8-0.9','0.9-1','Total images'])
    return mel_new_row,nev_new_row,proba_new_row