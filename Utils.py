import torch
import numpy as np
from torch.autograd import Variable

def to_tensor(ndarray,volatile = False, requires_grad=False):
    res = Variable(torch.from_numpy(ndarray),volatile=volatile, requires_grad=requires_grad).float()
    return res

def to_numpy(var):
    return var.data.numpy()

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)