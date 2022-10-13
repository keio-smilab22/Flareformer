"""
Train and eval functions used in main.py
"""

from typing import Dict, Tuple, Any
import torch
from utils.losses import Losser
from utils.statistics import Stat
from tqdm import tqdm
from torch.optim.adam import Adam
from torch.utils.data.dataloader import DataLoader


def train_epoch(model: torch.nn.Module,
                optimizer: Adam,
                train_dl: DataLoader,
                losser: Losser,
                stat: Stat) -> Tuple[Dict[str, Any], float]:
    """train one epoch"""
    model.train()
    losser.clear()
    stat.clear_all()
    for _, (x, y, _) in enumerate(tqdm(train_dl)):
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


def eval_epoch(model: torch.nn.Module,
               val_dl: DataLoader,
               losser: Losser,
               stat: Stat) -> Tuple[Dict[str, Any], float]:
    """evaluate the given model"""
    model.eval()
    losser.clear()
    stat.clear_all()
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
