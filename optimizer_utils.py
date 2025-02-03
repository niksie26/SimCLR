import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.optim import SGD
import pdb
import os
import shutil
import numpy as np
import warnings
import time
import torch.distributed as dist
import threading


class LARS(SGD):
    """
    Slight modification of LARC optimizer from https://github.com/NVIDIA/apex/blob/d74fda260c403f775817470d87f810f816f3d615/apex/parallel/LARC.py
    Matches one from SimCLR implementation https://github.com/google-research/simclr/blob/master/lars_optimizer.py
    Args:
        optimizer: Pytorch optimizer to wrap and modify learning rate for.
        trust_coefficient: Trust coefficient for calculating the adaptive lr. See https://arxiv.org/abs/1708.03888
    """

    def __init__(self,  param_groups, lr, momentum, trust_coefficient=0.001):
        super(LARS, self).__init__(param_groups, lr, momentum)
        self.trust_coefficient = trust_coefficient

    def step(self):
        with torch.no_grad():
            weight_decays = []
            for group in self.param_groups:
                # absorb weight decay control from optimizer
                weight_decay = group['weight_decay'] if 'weight_decay' in group else 0
                weight_decays.append(weight_decay)
                group['weight_decay'] = 0
                for p in group['params']:
                    if p.grad is None:
                        continue

                    if weight_decay != 0:
                        p.grad.data += weight_decay * p.data

                    param_norm = torch.norm(p.data)
                    grad_norm = torch.norm(p.grad.data)
                    adaptive_lr = 1.

                    if param_norm != 0 and grad_norm != 0 and group['layer_adaptation']:
                        adaptive_lr = self.trust_coefficient * param_norm / grad_norm

                    p.grad.data *= adaptive_lr

        super().step()
        # return weight decay control to optimizer
        for i, group in enumerate(self.param_groups):
            group['weight_decay'] = weight_decays[i]



class LinearLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, num_epochs, last_epoch=-1):
        self.num_epochs = max(num_epochs, 1)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        res = []
        for lr in self.base_lrs:
            res.append(np.maximum(lr * np.minimum(-self.last_epoch * 1. / self.num_epochs + 1., 1.), 0.))
        return res


class LinearWarmupAndCosineAnneal(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warm_up, T_max, last_epoch=-1):
        self.warm_up = int(warm_up * T_max)
        self.T_max = T_max - self.warm_up
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")

        if self.last_epoch == 0:
            return [lr / (self.warm_up + 1) for lr in self.base_lrs]
        elif self.last_epoch <= self.warm_up:
            c = (self.last_epoch + 1) / self.last_epoch
            return [group['lr'] * c for group in self.optimizer.param_groups]
        else:
            # ref: https://github.com/pytorch/pytorch/blob/2de4f245c6b1e1c294a8b2a9d7f916d43380af4b/torch/optim/lr_scheduler.py#L493
            le = self.last_epoch - self.warm_up
            return [(1 + np.cos(np.pi * le / self.T_max)) /
                    (1 + np.cos(np.pi * (le - 1) / self.T_max)) *
                    group['lr']
                    for group in self.optimizer.param_groups]


class BaseLR(torch.optim.lr_scheduler._LRScheduler):
    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

    
    
def get_optimizer_scheduler(opt, LR, weight_decay, lr_schedule, warmup, T_max, model, last_epoch, betas = (0.9, 0.999)):

    def exclude_from_wd_and_adaptation(name):
        if 'bn' in name:
            return True
        if opt == 'lars' and 'bias' in name:
            return True

    param_groups = [
        {
            'params': [p for name, p in model.named_parameters() if not exclude_from_wd_and_adaptation(name)],
            'weight_decay': weight_decay,
            'layer_adaptation': True,
        },
        {
            'params': [p for name, p in model.named_parameters() if exclude_from_wd_and_adaptation(name)],
            'weight_decay': 0.,
            'layer_adaptation': False,
        },
    ]

    if opt == 'sgd':
        optimizer = torch.optim.SGD(
            param_groups,
            lr=LR,
            momentum=0.9,
        )
    elif opt == 'adam':
        optimizer = torch.optim.Adam(
            param_groups,
            lr=LR,
            betas=betas
        )
    elif opt == 'lars':
        optimizer = LARS(
            param_groups,
            lr=LR,
            momentum=0.9,
        )
#         larc_optimizer = LARS(optimizer)
    else:
        raise NotImplementedError

    if lr_schedule == 'warmup-anneal':
        scheduler = LinearWarmupAndCosineAnneal(
            optimizer,
            warmup,
            T_max,
            last_epoch=last_epoch,
        )
    elif lr_schedule == 'linear':
        scheduler = LinearLR(optimizer, T_max, last_epoch=last_epoch)
    elif lr_schedule == 'const':
        scheduler = None
    else:
        raise NotImplementedError

#     if opt == 'lars':
#         optimizer = larc_optimizer

    return optimizer, scheduler
