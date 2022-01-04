## Addressing Class Imbalance with Hard Mining Loss (HMLoss)
_________________

This is the official implementation of HMLoss in the paper [Addressing Class Imbalance with Hard Mining Loss](https:) in Tensorflow.

### Dependency
The code is buil with following libraries
- [Tensorflow](https://www.tensorflow.org) 2.4
- [Tensorflow](https://www.tensorflow.org) 2.4
- [scikit-learn](https://scikit-learn.org/stable/)

### Dataset
- Imbalanced [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html). The original data will be downloaded and converted by `imbalancec_cifar.py`.
- ISIC2019 [ISIC2019](https://challenge2019.isic-archive.com/). The original data will be preprocessed and split by `preprocessing.py`.

### Training
We provide several training examples with this repo:

- To train the HMLoss baseline on long-tailed imbalance with ratio of 100

```bash
python cifar_train.py --gpu 0 --imb_type exp --imb_factor 0.01 --loss_type CE --train_rule None
```


### Reference

If you find our paper and repo useful, please cite as

```
@inproceedings{
}
