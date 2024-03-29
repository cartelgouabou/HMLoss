## Rethinking decoupled training with bag of tricks for long-tailed recognition
_________________

This is the official implementation of HMLoss in the paper [Rethinking decoupled training with bag of tricks for long-tailed recognition](https://hal.archives-ouvertes.fr/hal-03826758/) in Pytorch.
Find the french version of the paper published in GRETSI 2022 [there](https://hal.archives-ouvertes.fr/hal-03725510).
### Abstract figure

![Alt text](ressources/images/abstract_figure.png?raw=true "HMLoss")
### Dependency
The code is build with following main libraries
- [Pytorch](https://www.tensorflow.org) 1.11.0
- [Numpy](https://numpy.org/) 
- [Pandas](https://pandas.pydata.org/)
- [Sklearn](https://scikit-learn.org/stable/)
- [Matlab](https://ch.mathworks.com/fr/products/matlab.html)

You can install all dependencies with requirements.txt following the command:
```bash
pip install -r requirements.txt 
```


### Dataset
- Imbalanced [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html). The original data will be downloaded and converted by `/cifar_implementation/imbalance_cifar.py`.
- ISIC2019 [ISIC2019](https://challenge2019.isic-archive.com/). The original data will be preprocessed by `/isic2019_implementation/preprocessing/preprocessImageConstancy.m`and split by `/isic2019_implementation/preprocessing/train_valid_split_task.py`.


# On cifar repo
- To train the HMLoss baseline on long-tailed imbalance with ratio of 200 

```bash
python train_cifar.py --loss_function 'HML1' --weighting_type CS --dataset_name 'cifar100' --imb_type 'exp' --imb_ratio 0.02 --gpu 0 
```

# On isic repo
- To train the HMLoss baseline on 2-class version of isic2019 for melanoma versus nevi classification

```bash
python train_isic.py --loss_type 'HML1' --delta 10000   
```
# On inaturalist repo
- To train the HMLoss baseline on inaturalist 2018 dataset

```bash
python train_inat.py --loss_type 'HML1' --delta 10000   
```

### Reference

If you find our paper and repo useful, please cite as

```
@inproceedings{foahom2022rethinking,
  title={Rethinking decoupled training with bag of tricks for long-tailed recognition},
  author={Foahom Gouabou, Arthur Cartel and Iguernaissi, Rabah and Damoiseaux, Jean Luc and Moudafi, Abdellatif and Merad, Djamal},
  booktitle={2022 Digital Image Computing: Techniques and Applications (DICTA)},
  pages={1--8},
  year={2022},
  organization={IEEE}
}
