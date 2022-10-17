"""学習向けのユーティリティモジュール"""
import math
from argparse import Namespace
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE


def fix_seed(seed: int) -> None:
    """Fix seed.

    Args:
        seed (int): seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def inject_args(args: Namespace, target: Dict[str, Any]) -> Namespace:
    """Inject args.

    Args:
        args (Namespace): argparser.args
        target (Dict[str, Any]): target dict

    Returns:
        Namespace: args
    """
    for key, value in target.items():
        args.__setattr__(key, value)
    return args


def adjust_learning_rate(optimizer, current_epoch, epochs, lr, args):  # optimizerの内部パラメタを直接変えちゃうので注意
    """Decay the learning rate with half-cycle cosine after warmup."""
    min_lr = 0
    if current_epoch < args.warmup_epochs:
        lr = lr * current_epoch / args.warmup_epochs
    else:
        theta = math.pi * (current_epoch - args.warmup_epochs) / (epochs - args.warmup_epochs)
        lr = min_lr + (lr - min_lr) * 0.5 * (1.0 + math.cos(theta))

    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr

    return lr


def visualize_3D_tSNE(embedded: torch.Tensor, labels: torch.Tensor):
    """
    Visualize 3d embedded vector using t-SNE.
    """
    if len(labels.shape) != 1:  # if labels are one-hot
        labels = labels.argmax(axis=1)

    tsne = TSNE(n_components=3, init="pca", random_state=0, perplexity=30, n_iter=1000)
    reduced = tsne.fit_transform(embedded.cpu().numpy())

    labels_np = labels.cpu().numpy()
    N = np.max(labels_np) + 1
    fig = plt.figure(figsize=(10, 10)).gca(projection="3d")
    for i in range(N):
        target = reduced[labels_np == i]
        fig.scatter(target[:, 0], target[:, 1], target[:, 2], label=str(i), alpha=0.5)
    fig.legend(bbox_to_anchor=(1.02, 0.7), loc="upper left")
    plt.show()
