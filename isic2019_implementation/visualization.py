#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 15:00:22 2021

@author: arthur
"""
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 
def learning_curve(history,checkpoint_root,title_text):
    cols = ['{}'.format(col) for col in ['Model accuracy '+title_text,'Model loss '+title_text]]
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(40, 20))
    for ax, col in zip(axes, cols):
            ax.set_title(col,fontsize=20)
    # Historique des précisions
    axes[0].plot(history.history['accuracy'],label='train') 
    axes[0].plot(history.history['val_accuracy'],label='test')
    axes[0].legend(loc="upper right",fontsize=20)
    axes[0].set_xlabel('epoch',fontsize=20)
    axes[0].set_ylabel('accuracy',fontsize=20)
    
    # Historique des erreurs
    axes[1].plot(history.history['loss'],label='train')   
    axes[1].plot(history.history['val_loss'],label='test')
    axes[1].legend(loc="upper right",fontsize=20)
    axes[1].set_xlabel('epoch',fontsize=20)
    axes[1].set_ylabel('loss',fontsize=20)
    plt.savefig(checkpoint_root+'/history_case_'+title_text+'_plot.png')
    fig.tight_layout()
    plt.close()
   
def cyclical_plot(clr,checkpoint_root,title_text):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(40, 20))
    ax.set_title("Cycle Policy "+title_text,fontsize=20)
    ax.plot(clr.history['iterations'], clr.history['lr'])
    ax.set_xlabel('Training Iterations',fontsize=20)
    ax.set_ylabel('Learning Rate',fontsize=20)
    plt.savefig(checkpoint_root+'/Cycle Policy_'+title_text+'_plot.png')
    plt.close()
