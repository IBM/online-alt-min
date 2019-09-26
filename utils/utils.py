from collections import MutableMapping
from random import randint
import os
import time
import copy
import torch


class ddict(object):
    '''
    dd = ddict(lr=[0.1, 0.2], n_hiddens=[100, 500, 1000], n_layers=2)
    '''
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __getitem__(self, item):
        return self.__dict__.__getitem__(item)

    def __repr__(self):
        return str(self.__dict__)


def get_devices(cuda_device="cuda:0", seed=1):
    device = torch.device(cuda_device)
    torch.manual_seed(seed)
    # Multi GPU?
    num_gpus = torch.cuda.device_count()
    if device.type != 'cpu':
        print('\033[93m'+'Using CUDA,', num_gpus, 'GPUs\033[0m')
        torch.cuda.manual_seed(seed)
    return device, num_gpus

