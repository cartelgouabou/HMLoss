## Addressing Class Imbalance with Hard Mining Loss (HMLoss)
_________________

This is the official implementation of HMLoss in the paper [Addressing Class Imbalance with Hard Mining Loss](https:) in Pytorch.
### Abstract figure

![Alt text](ressources/images/abstract_figure.png?raw=true "HMLoss")
### Dependency
The code is build with following libraries
- [Pytorch](https://www.tensorflow.org) 1.11.0
- [Numpy](https://numpy.org/) 
- [Pandas](https://pandas.pydata.org/)
- [Sklearn](https://scikit-learn.org/stable/)
- [Matlab](https://ch.mathworks.com/fr/products/matlab.html)


### Dataset
- Imbalanced [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html). The original data will be downloaded and converted by `/cifar_implementation/imbalance_cifar.py`.
- ISIC2019 [ISIC2019](https://challenge2019.isic-archive.com/). The original data will be preprocessed by `/isic2019_implementation/preprocessing/preprocessImageConstancy.m`and split by `/isic2019_implementation/preprocessing/train_valid_split_task.py`.


# On cifar repo
- To train the HMLoss baseline on long-tailed imbalance with ratio of 200 

```bash
python train_cifar.py --loss_function 'HML' --dataset_name 'cifar100' --imb_type 'exp' --imb_ratio 0.02  
```

- To generate result once the models are trained, example with the HMLoss baseline on long-tailed imbalance with ratio of 100

```bash
python result_analysis.py --dataset 'cifar10' --loss_function 'hmld10000000g05a75' --imb_type 'exp' --imb_ratio 0.01 
```

# On isic repo
- To train the HMLoss baseline on 2-class version of isic2019 for melanoma versus nevi classification

```bash
python train_isic.py --loss_type 'HML' --delta 10000   
```
# On inaturalist repo
- To train the HMLoss baseline on inaturalist 2018 dataset

```bash
python train_inat.py --loss_type 'HML' --delta 10000   
```

### Reference

If you find our paper and repo useful, please cite as

```
@inproceedings{
}
