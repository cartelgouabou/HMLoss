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
from resnet_cifar import ResNet32
from imbalance_cifar import IMBALANCECIFAR
from model_trainer import model_compiler,train_model 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from backup import LinearWarmUpScheduler
from visualization import learning_curve
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
parser = argparse.ArgumentParser(description='Tensorflow Cifar Training')
parser.add_argument('--dataset','--dataset_name', default='cifar10', help='dataset setting',dest='dataset_name')
parser.add_argument('--loss_function', default="ce", type=str, help='loss function name',dest='case_loss_fn')
parser.add_argument('--loss_type', default="softmax", type=str,choices=['softmax','logits','focal'], help='indicate the activation function to use, for focal loss use "focal", for LDAM, csLDAM and CBLDAM use "logits", and for the rest use "softmax"')
parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
parser.add_argument('--imb_ratio','--ir', default=0.01, type=float, help='imbalance ratio, 1 for balanced, 0.1 or 0.01',dest='ir')
parser.add_argument('--lr', '--learning-rate','--init-lr', default=0.1, type=float, help='define the initial learning rate. The learning rate will decrease during training. For more details check the backup implementation in  this file',dest='init_lr')
parser.add_argument('-b','--batch_size', default=128, type=int, help='define batch size')
parser.add_argument('--epochs', default=200, type=int,help='number of total epochs to run')
parser.add_argument('--num_runs', default=3, type=int, help='number of runs to launch ')
parser.add_argument('--wd', '--weight-decay', default=0.0002, type=float,
                    metavar='W', help='weight decay fro convolutions (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--kernel-initializer', '--k-init', default='he_normal', type=str,help='define kernel initializer (default: he_normal)',
                    dest='kernel_initializer')


args = parser.parse_args()
for run in range(args.num_runs):
    imbalanceData = IMBALANCECIFAR(dataset_name=args.dataset_name,  imb_type=args.imb_type, imb_factor=args.ir)
    x_train=imbalanceData.train_images
    y_train=imbalanceData.train_labels
    x_test=imbalanceData.test_images
    y_test=imbalanceData.test_labels
    num_class=imbalanceData.cls_num
    y_train= to_categorical(y_train, num_classes=num_class)
    y_test= to_categorical(y_test, num_classes=num_class)
    img_num_per_class=imbalanceData.get_cls_num_list()
    train_size=len(x_train)
    input_shape = list(x_train.shape[1:])   
    train_datagen = ImageDataGenerator(rescale=1./255,
            rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True)  # randomly flip images
    valid_datagen = ImageDataGenerator(rescale=1./255)
    checkpoint_root=root_path+'/'+args.case_loss_fn+'/checkpoint/'
    if not os.path.exists(checkpoint_root+'/best_weights'):
                os.makedirs(checkpoint_root+'/best_weights')
    if not os.path.exists(checkpoint_root+'/training_history'):
                os.makedirs(checkpoint_root+'/training_history')
    print('curent case:', args.case_loss_fn)
    print('curent IR:', args.ir)
    print('curent run:', run)
    model=ResNet32(input_shape, num_class, 'ResNet-32',args.loss_type,args.weight_decay,args.kernel_initializer)
    filenames=args.dataset_name+'_'+args.imb_type+'_'+args.case_loss_fn+'_ir_'+str(args.ir)+'_run'+str(run)
    model_compiler(model,args.case_loss_fn,num_class,img_num_per_class)
    callback = LinearWarmUpScheduler(args.init_lr,checkpoint_root,filenames,args.epochs) 
    resultat=train_model(model,args.case_loss_fn,x_train,y_train,x_test,y_test,callback,args.batch_size,args.epochs,train_datagen,valid_datagen,num_class,img_num_per_class)
    title_text=args.dataset_name+'_'+args.imb_type+'_'+args.case_loss_fn+'_ir_'+str(args.ir)+'_run'+str(run)
    learning_curve(resultat,checkpoint_root+'/training_history',title_text)
     