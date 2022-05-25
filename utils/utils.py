import numpy as np
import torch
import math


def fix_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def inject_args(args, target):
    for key, value in target.items():
        args.__setattr__(key, value)
    return args


def adjust_learning_rate(optimizer, current_epoch, epochs, lr, args):  # optimizerの内部パラメタを直接変えちゃうので注意
    """Decay the learning rate with half-cycle cosine after warmup"""
    min_lr = 0
    if current_epoch < args.warmup_epochs:
        lr = lr * current_epoch / args.warmup_epochs
    else:
        theta = math.pi * (current_epoch - args.warmup_epochs) / (epochs - args.warmup_epochs)
        lr = min_lr + (lr - min_lr) * 0.5 * (1. + math.cos(theta))

    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr

    return lr
