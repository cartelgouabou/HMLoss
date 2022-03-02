# # Creating Train / Val / Test folders (One time use)
import os
import numpy as np
import shutil
root_dir = '/media/arthur/Data/PROJET/Class_imbalance/skin_cancer/case_isic2019/BASE' # data root path
root_source = '/media/arthur/Data/PROJET/HYPER_ARTICLE/BASE/ISIC2019_CROP_CC/' # data root path
classes = ['NEV','MEL'] #total labels 

train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

for classe in classes:
        if classe=='MEL':
            cls='MAL'
        else:
            cls='BEN'
        
        path_dir=root_dir+'/'
 	    # Creating partitions of the data after shuffeling
        src=root_source + classe # Folder to copy images from
        allFileNames = os.listdir(src)
        np.random.shuffle(allFileNames)
        train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                          [int(len(allFileNames)* train_ratio),int(len(allFileNames)* (train_ratio+val_ratio))]) 
        train_FileNames = [src+'/'+ name for name in train_FileNames.tolist()]
        val_FileNames = [src+'/' + name for name in val_FileNames.tolist()]
        test_FileNames = [src+'/' + name for name in test_FileNames.tolist()]

        print('Total images: ', len(allFileNames))
        print('Training: ', len(train_FileNames))
        print('Validation: ', len(val_FileNames))
        print('Testing: ', len(test_FileNames))
        
        # Copy-pasting images
        if not os.path.exists(path_dir+'TRAIN/'+cls):
                os.makedirs(path_dir+'TRAIN/'+cls)
        for name in train_FileNames:
            shutil.copy(name, path_dir +'TRAIN/' + cls)
            
        if not os.path.exists(path_dir+'VALID/'+cls):
                 os.makedirs(path_dir+'VALID/'+cls)
        for name in val_FileNames:
             shutil.copy(name, path_dir +'VALID/' + cls)
            
        if not os.path.exists(path_dir+'TEST/'+cls):
                os.makedirs(path_dir+'TEST/'+cls)
        for name in test_FileNames:
            shutil.copy(name, path_dir +'TEST/' + cls)
            
