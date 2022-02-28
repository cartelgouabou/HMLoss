  # -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 18:58:52 2020

@author: arthu
"""

from __future__ import print_function, division
import argparse
import os
import tensorflow as tf
#L'étape de préparation de données se fait manuellement
# from tensorflow import set_random_seed
# set_random_seed(0)
from numpy.random import seed
seed(0)
from glob import glob  #Module permetant de faire une liste de chemin ayant un nom ou une caractéristiques rechercher
from utilities import read_and_preprocess
from models import EfficientNetBias,EfficientNet
from model_trainer import model_compiler,train_model 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from backup import CyclicLR,BaccCallback
from visualization import learning_curve,cyclical_plot
from tensorflow.compat.v1 import ConfigProto         #resolve the train_function error with Nvidia RTX GPU
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
if tf.test.gpu_device_name(): 
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")
gpus = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

root_path=os.getcwd()
parser = argparse.ArgumentParser(description='Tensorflow Isic Training')
parser.add_argument('--class_1_name', default='MAL', help='specified the name of class1 as it is write on the repository',dest='class_1_name')
parser.add_argument('--loss_function', default="bce", type=str, help='loss function name',dest='case_loss_fn')
parser.add_argument('--max_lr', '--learning-rate', default=0.001, type=float, help='define the maximum learning rate during cyclical learning. For more details check the backup implementation in  this file',dest='max_lr')
parser.add_argument('-b','--batch_size', default=128, type=int, help='define batch size')
parser.add_argument('-ratio_freeze', default=64, type=int, help='percentage of earlier layers to freeze')
parser.add_argument('--epochs', default=100, type=int,help='number of total epochs to run')
parser.add_argument('--step_size', default=2000, type=int,help='specified step size for cyclical learning')
parser.add_argument('--delay', default=15, type=int,help='number of epoch to patience before early stopping the training if no improvement of balanced accuracy')
parser.add_argument('--num_runs', default=3, type=int, help='number of runs to launch ')


args = parser.parse_args()

data_root=root_path+'/BASE/'   #end with /
checkpoint_root=root_path+'/checkpoint/'
image_size = [300,300]

task_name='ISIC2019'


for run in range(args.num_runs):
    print('curent case:', args.case_loss_fn)
    print('curent run:', run)
    #Importation et préparation des données
    #Direction de la base de données
    train_path = data_root+'/TRAIN'
    valid_path =data_root+'/VALID'
    # évalue le nombre d'image dans chaque dossier
    train_set = glob(train_path + '/*/*.JPG')
    valid_set = glob(valid_path + '/*/*.JPG')
    [x_valid,y_valid,train_filenames]=read_and_preprocess(valid_path,image_size,preprocess_input)
    train_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)
    valid_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)
    train_gen = train_datagen.flow_from_directory(train_path,
                                          target_size = image_size,   #taille des nouvelles images
                                          shuffle=True, #Permet de mélanger aléatoirement le jeu de donnée permettant ainsi d'améliorer la qualité du modèle et ainsi que ses performances
                                          batch_size = args.batch_size,          #mise à jout des poids apres des lots d'observation
                                          class_mode = 'binary')
    valid_gen = valid_datagen.flow_from_directory(valid_path,
                                    target_size = image_size,
                                    shuffle=False,
                                    batch_size = args.batch_size,
                                    class_mode = 'binary')
    # évalue le nombre de classes
    numClass = glob(train_path + '/*')
    nclass1=len(glob(train_path+'/'+args.class_1_name+'/*'))
    nclass0=len(glob(train_path + '/*/*.JPG'))-nclass1
    img_num_per_class=[nclass0,nclass1]
    checkpoint_root=root_path+'/checkpoint/'+args.case_loss_fn
    if not os.path.exists(checkpoint_root+'/cnn_architecture'):
            os.makedirs(checkpoint_root+'/cnn_architecture')
    if not os.path.exists(checkpoint_root+'/best_weights'):
            os.makedirs(checkpoint_root+'/best_weights')
    if not os.path.exists(checkpoint_root+'/training_history'):
            os.makedirs(checkpoint_root+'/training_history')
    path_history=checkpoint_root+'/training_history/'
    path_weights=checkpoint_root+'/best_weights/'
    path_weights_load=''
    load_w=False
    base_lr = args.max_lr/100
    model=EfficientNetBias(args.ratio_freeze,path_weights_load,load_w,'sigmoid')
    if (args.case_loss_fn=='hinge') | (args.case_loss_fn=='cshinge'):
        model=EfficientNetBias(args.ratio_freeze,path_weights_load,load_w,'tanh')
    elif (args.case_loss_fn=='LDAM') | (args.case_loss_fn=='LDAM-RW'):
        model=EfficientNet(args.ratio_freeze,path_weights_load,load_w)
    bcb=BaccCallback(x_valid,y_valid,path_history,path_weights,task_name,args.case_loss_fn,'run'+str(run),args.delay,args.epochs)
    clr = CyclicLR(base_lr=base_lr, max_lr=args.max_lr,step_size=args.step_size, mode='triangular2')
    model_compiler(model,args.case_loss_fn,img_num_per_class)
    resultat=train_model(model,args.case_loss_fn,train_gen,valid_gen,bcb,clr,len(train_set),len(valid_set),args.batch_size,args.epochs,img_num_per_class)
    title_text=task_name+'_'+args.case_loss_fn+'_run'+str(run)
    cyclical_plot(clr,checkpoint_root+args.case_loss_fn+'/training_history',title_text)
    learning_curve(resultat,checkpoint_root+args.case_loss_fn+'/training_history',title_text)
           
     