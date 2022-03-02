clc;
clear all;
close all;
%R=img(:,:,3);
%pathTran1 = 'D:\PROJET\DERMA_ARTICLE\base\melVSnev_cc\sansHalo\tran';
pathTran1 = 'D:\PROJET\DERMA_ARTICLE\base\melVSnev_cc\avecHalo\tran';

%pathTest1 = 'D:\PROJET\DERMA_ARTICLE\base\melVSnev_cc\sansHalo\test';
pathTest1 = 'D:\PROJET\DERMA_ARTICLE\base\skinan_cropt';



dataDir1 =  fullfile('D:','PROJET','DERMA_ARTICLE','base','skinan','MAL');
data1 = datastore(dataDir1);
path1=char(data1.Files(1));
img1=imread(path1);
pos1=41
path1(pos1:end)

%Test
for i = 1:2
    if i==1
        temp='MAL';
    else
        temp='BEN';
    end
    dataDir =  fullfile('D:','PROJET','DERMA_ARTICLE','base','skinan',temp);
    data = datastore(dataDir);
    num = numel(data.Files); 
    for j = 1: num
        path=char(data.Files(j));
        img=imread(path);
        filename=path(pos1:end);
        imgTreat = cropCenterISIC19(img);
        %imgTreat = colorConstancy(img, 'gray world seg',2);
        fullFileName = fullfile(pathTest1,temp,filename);
        imwrite(imgTreat,fullFileName);
    end
end

%tran
for i = 1:2
    if i==1
        temp='mel';
    else
        temp='mnv';
    end
    dataDir =  fullfile('D:','PROJET','DERMA_ARTICLE','base','melVSnev','avecHalo','tran',temp);
    data = datastore(dataDir);
    num = numel(data.Files); 
    for j = 1: num
        path=char(data.Files(j));
        img=imread(path);
        filename=path(pos2:end);
        %imgTreat = cropCenter(img, 450);
        imgTreat = colorConstancy(img, 'gray world seg',2);
        fullFileName = fullfile(pathTran1,temp,filename);
        imwrite(imgTreat,fullFileName);
    end
end
