
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

num_classes = 2

def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values 
    return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=1.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), self.gamma)
    
    
    

def softmax_hard_mining_loss(input_values,  delta):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    eps=1e-10
    p=torch.clip(p,eps,1.-eps)
    loss = ((torch.sin((p)*np.pi)/((p)*np.pi))-( (torch.exp(-delta*p))*(torch.sin((p)*delta*np.pi)/((p)*delta*np.pi))) )  * input_values * 10
    return loss.mean()

class SoftmaxHardMiningLoss(nn.Module):
    def __init__(self, weight=None,delta=10000000000):
        super(SoftmaxHardMiningLoss, self).__init__()
        assert delta >= 0
        self.weight = weight
        self.delta=delta


    def forward(self, input, target):
        return softmax_hard_mining_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight),self.delta)
class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)
