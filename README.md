## Addressing Class Imbalance with Hard Mining Loss (HMLoss)
_________________

This is the official implementation of HMLoss in the paper [Addressing Class Imbalance with Hard Mining Loss](https:) in Tensorflow.
### Abstract figure

![Alt text](ressources/images/abstract_figure.png?raw=true "HMLoss")
### Dependency
The code is build with following libraries
- [Tensorflow](https://www.tensorflow.org) 2.4
- [Numpy](https://numpy.org/) 
- [Pandas](https://pandas.pydata.org/)
- [Sklearn](https://scikit-learn.org/stable/)

### Typical use
Standalone usage:
    
    >>> y_true = [[1,0,0], [1,0,0], [0,1,0],[0,0,1]]
    >>> y_pred = [[0.8,0.1,0.1], [0.9,0.1,0.0], [0.1,0.1,0.8],[0.1,0.2,0.7]]
    >>> hml = CategoricalHardMiningLoss(alpha=0.75,delta=10000000,gamma=1)
    >>> hml(y_true,y_pred).numpy()
    0.46121484

```python
 # Typical tf.keras API usage
    import tensorflow as tf
    from losses import CategoricalHardMiningLoss

    model = tf.keras.Model(...)
    model.compile(
        optimizer=...,
        loss=CategoricalHardMiningLoss(alpha=.75, gamma=1, detlta=10000000))   # Used here like a tf.keras loss
        metrics=...,
    )
    history = model.fit(...)
```


### Dataset
- Imbalanced [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html). The original data will be downloaded and converted by `/cifar_implementation/imbalance_cifar.py`.
- ISIC2019 [ISIC2019](https://challenge2019.isic-archive.com/). The original data will be preprocessed by `/isic2019_implementation/preprocessing/preprocessImageConstancy.m`and split by `/isic2019_implementation/preprocessing/train_valid_split_task.py`.

### Training
We provide several training examples with this repo. Please refer to `codification_of_loss_function.txt` to know the coding of the cost function to use

# On cifar repo
- To train the HMLoss baseline on long-tailed imbalance with ratio of 100 

```bash
python train_cifar.py --loss_function 'hmld10000000g05a75' --dataset_name 'cifar10' --loss_type 'softmax' --imb_type 'exp' --imb_ratio 0.01  
```

- To generate result once the models are trained, example with the HMLoss baseline on long-tailed imbalance with ratio of 100

```bash
python result_analysis.py --dataset 'cifar10' --loss_function 'hmld10000000g05a75' --imb_type 'exp' --imb_ratio 0.01 
```

# On isic repo
- To train the HMLoss baseline on 2-class version of isic2019 for melanoma versus nevi classification

```bash
python train_isic.py --loss_function 'hmld100000g05a75'   
```


### Reference

If you find our paper and repo useful, please cite as

```
@inproceedings{
}
