clc;
clear all;
close all;
%R=img(:,:,3);
pathSource = 'D:\PROJET\Class_imbalance\original_dataset';

pathDest = 'D:\PROJET\Class_imbalance\dataset';



dataDir1 =  fullfile('D:','PROJET','Class_imbalance','original_dataset','MAL');
data1 = datastore(dataDir1);
path1=char(data1.Files(1));
img1=imread(path1);
pos1=41
path1(pos1:end)

for i = 1:2
    if i==1
        temp='MAL';
    else
        temp='BEN';
    end
    dataDir =  fullfile('D:','PROJET','Class_imbalance','original_dataset',temp);
    data = datastore(dataDir);
    num = numel(data.Files); 
    for j = 1: num
        path=char(data.Files(j));
        img=imread(path);
        filename=path(pos1:end);
        imgTreat = cropCenterISIC19(img);
        imgTreat = colorConstancy(img, 'gray world seg',2);
        fullFileName = fullfile(pathTest1,temp,filename);
        imwrite(imgTreat,fullFileName);
    end
end
