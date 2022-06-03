"""
Train and eval functions used in main.py
"""

import numpy as np
import torch
from models.model import FlareFormer
from utils.losses import Losser
from utils.statistics import Stat
from utils.utils import adjust_learning_rate

from tqdm import tqdm
from argparse import Namespace
from typing import Dict, Tuple, Any
from torch.optim.adam import Adam
from torch.utils.data.dataloader import DataLoader


def train_epoch(model: FlareFormer,
                optimizer: Adam,
                train_dl: DataLoader,
                epoch: int,
                lr: float,
                args: Namespace,
                losser: Losser,
                stat: Stat) -> Tuple[Dict[str, Any], float]:
    """train one epoch"""
    model.train()
    losser.clear()
    for _, (x, y, _) in enumerate(tqdm(train_dl)):
        if not args.without_schedule:
            adjust_learning_rate(optimizer, epoch, args.dataset["epochs"], lr, args)
        optimizer.zero_grad()

        imgs, feats = x
        imgs, feats = imgs.cuda().float(), feats.cuda().float()
        output, _ = model(imgs, feats)
        gt = y.cuda().to(torch.float)
        loss = losser(output, gt)
        loss.backward()
        optimizer.step()
        stat.collect(output, y)

    score = stat.aggregate("train")
    return score, losser.get_mean_loss()


def eval_epoch(model: FlareFormer,
               val_dl: DataLoader,
               losser: Losser,
               args: Namespace,
               stat: Stat) -> Tuple[Dict[str, Any], float]:
    """evaluate the given model"""
    model.eval()
    losser.clear()
    with torch.no_grad():
        for _, (x, y, _) in enumerate(tqdm(val_dl)):
            imgs, feats = x
            imgs, feats = imgs.cuda().float(), feats.cuda().float()
            output, _ = model(imgs, feats)
            gt = y.cuda().to(torch.float)
            _ = losser(output, gt)
            stat.collect(output, y)

    score = stat.aggregate("valid")
    return score, losser.get_mean_loss()
