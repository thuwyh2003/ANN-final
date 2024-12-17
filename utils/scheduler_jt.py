#ANN-Final
import logging
import math
import jittor.lr_scheduler as jt_scheduler

# from torch.optim.lr_scheduler import LambdaLR

logger = logging.getLogger(__name__)

class WarmupLinearSchedule(object):
    def __init__(self, optimizer, warmup_steps, t_total,last_epoch=-1):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.last_epoch = last_epoch
        self.base_lrs = [group['initial_lr'] for group in optimizer.param_groups]

    def step(self):
        self.last_epoch += 1
        # values = self.get_lr()
        # for param_group, lr in zip(self.optimizer.param_groups, values):
        #     param_group['lr'] = lr
    
    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))

    def get_lr(self):
        return [base_lr * self.lr_lambda(self.last_epoch) for base_lr in self.base_lrs]
    
class WarmupCosineSchedule(object):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        self.optimizer = optimizer
        # self.base_lrs = [group['initial_lr'] for group in optimizer.param_groups]        #  没看懂什么意思
        self.base_lrs = [0.03 for i in range(self.t_total)]      #WYH
        self.last_epoch = last_epoch

    def step(self):
        self.last_epoch += 1
        # values = self.get_lr()
        # for param_group, lr in zip(self.optimizer.param_groups, values):
        #     param_group['lr'] = lr
    
    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))
    
    def get_lr(self):
        return [base_lr * self.lr_lambda(self.last_epoch) for base_lr in self.base_lrs]
