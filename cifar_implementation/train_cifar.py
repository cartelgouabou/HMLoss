#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 10:02:32 2022

@author: arthur
"""

import argparse
import os
import random
import time
import warnings
import numpy as np
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from utils import *
from trainer import *
from opts import parser
args = parser.parse_args()
if args.seed is not None:
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    np.random.seed(args.seed)
    random.seed(args.seed)

if args.gpu is not None:
    warnings.warn('You have chosen a specific GPU.')

ngpus_per_node = torch.cuda.device_count()
run_list=[]
for run in range(args.num_runs):
    run_list.append('run'+str(run))
args.run_list = run_list

if args.imb_type=='both':
    args.imb_type=['exp','step']
else:
    args.imb_type=[args.imb_type]

print('curent case:')
print(args.loss_type)
print('List of run:')
print(args.run_list)
print('List of imbalance ratio:')
print(args.ir_list)

for ir in args.ir_list:
 for imb_type in args.imb_type:
    for run in run_list:
        best_acc1 = 0
        print('curent IR:')
        print(ir)
        print('curent run:')
        print(run)
        if 'HML' in args.loss_type:
            args.store_name = '_'.join([args.dataset, imb_type, str(ir), args.arch, args.loss_type, 'W' ,args.weighting_type,'D',str(args.delta),run])
        else:
            args.store_name = '_'.join([args.dataset, imb_type, str(ir), args.arch, args.loss_type, 'W' ,args.weighting_type])
        prepare_folders(args)
        main_worker(args.gpu, ngpus_per_node, args,ir,best_acc1,imb_type)
            


# if __name__ == '__main__':
#     main()
