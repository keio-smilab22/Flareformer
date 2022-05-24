import numpy as np
import torch

def fix_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def inject_args(args,target):
    for key, value in target.items():
        args.__setattr__(key, value)
    return args
