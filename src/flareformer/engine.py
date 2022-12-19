"""
Train and eval functions used in main.py .
"""

from typing import Any, Dict, Tuple

import torch
from torch.optim.adam import Adam
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from utils.losses import Losser
from utils.statistics import Stat


def train_epoch(
    model: torch.nn.Module, optimizer: Adam, train_dl: DataLoader, losser: Losser, stat_1h_12h: Stat, stat_12h_24h: Stat
) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
    """Train one epoch."""
    model.train()
    losser.clear()
    stat_1h_12h.clear_all()
    stat_12h_24h.clear_all()
    for _, (x, y, _) in enumerate(tqdm(train_dl)):
        optimizer.zero_grad()
        imgs, feats = x
        imgs, feats = imgs.cuda().float(), feats.cuda().float()
        output, _ = model(imgs, feats)
        output_1h_12h, output_12h_24h = output
        y_1h_12h, y_12h_24h = y
        gt_1h_12h = y_1h_12h.cuda().to(torch.float)
        loss_1h_12h = losser(output, gt_1h_12h)
        gt_12h_24h = y_12h_24h.cuda().to(torch.float)
        loss_12h_24h = losser(output, gt_12h_24h)
        loss = loss_1h_12h + loss_12h_24h
        loss.backward()
        optimizer.step()
        stat_1h_12h.collect(output_1h_12h, y_1h_12h)
        stat_12h_24h.collect(output_12h_24h, y_12h_24h)

    score_1h_12h = stat_1h_12h.aggregate("train_1h_12h")
    score_12h_24h = stat_12h_24h.aggregate("train_12h_24h")
    return score_1h_12h, score_12h_24h, losser.get_mean_loss()


def eval_epoch(
    model: torch.nn.Module, val_dl: DataLoader, losser: Losser, stat_1h_12h: Stat, stat_12h_24h: Stat, test: bool=False
) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
    """Evaluate the given model."""
    model.eval()
    losser.clear()
    stat_1h_12h.clear_all()
    stat_12h_24h.clear_all()
    with torch.no_grad():
        for _, (x, y, _) in enumerate(tqdm(val_dl)):
            imgs, feats = x
            imgs, feats = imgs.cuda().float(), feats.cuda().float()
            output, _ = model(imgs, feats)
            output_1h_12h, output_12h_24h = output
            y_1h_12h, y_12h_24h = y
            gt_1h_12h = y_1h_12h.cuda().to(torch.float)
            _ = losser(output, gt_1h_12h)
            gt_12h_24h = y_12h_24h.cuda().to(torch.float)
            _ = losser(output, gt_12h_24h)
            stat_1h_12h.collect(output_1h_12h, y_1h_12h)
            stat_12h_24h.collect(output_12h_24h, y_12h_24h)

    if test:
        score_1h_12h = stat_1h_12h.aggregate("test_1h_12h")
        score_12h_24h = stat_12h_24h.aggregate("test_12h_24h")

    else:
        score_1h_12h = stat_1h_12h.aggregate("val_1h_12h")
        score_12h_24h = stat_12h_24h.aggregate("val_12h_24h")

    return score_1h_12h, score_12h_24h, losser.get_mean_loss()
