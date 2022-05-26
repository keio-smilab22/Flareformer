"""
Train and eval functions used in main.py
"""

import numpy as np
import torch
from models.model import FlareFormer
from utils.losses import Losser
from utils.eval_utils import calc_score
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
                losser: Losser) -> Tuple[Dict[str, Any], float]:
    """train one epoch"""
    model.train()
    predictions = []
    observations = []
    train_loss = 0
    n = 0
    for _, (x, y, idx) in enumerate(tqdm(train_dl)):
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

        train_loss += (loss.detach().cpu().item() * imgs.shape[0])
        n += imgs.shape[0]

        for pred, o in zip(output.cpu().detach().numpy().tolist(),
                           y.detach().numpy().tolist()):
            predictions.append(pred)
            observations.append(np.argmax(o))

    score = calc_score(predictions, observations,
                       args.dataset["climatology"])
    score = calc_test_score(score, "train")

    return score, train_loss / n


def eval_epoch(model: FlareFormer,
               val_dl: DataLoader,
               losser: Losser,
               args: Namespace) -> Tuple[Dict[str, Any], float]:
    """evaluate the given model"""
    model.eval()
    predictions = []
    observations = []
    valid_loss = 0
    n = 0
    with torch.no_grad():
        for _, (x, y, idx) in enumerate(tqdm(val_dl)):
            imgs, feats = x
            imgs, feats = imgs.cuda().float(), feats.cuda().float()
            output, _ = model(imgs, feats)
            gt = y.cuda().to(torch.float)
            loss = losser(output, gt)
            valid_loss += (loss.detach().cpu().item() * imgs.shape[0])
            n += imgs.shape[0]
            for pred, o in zip(output.cpu().numpy().tolist(),
                               y.numpy().tolist()):
                predictions.append(pred)
                observations.append(np.argmax(o))
        score = calc_score(predictions, observations,
                           args.dataset["climatology"])
        score = calc_test_score(score, "valid")
    return score, valid_loss / n


def calc_test_score(score: Dict[str, Any], label: str) -> Dict[str, Any]:
    """Return dict with key of label"""
    test_score = {}
    for k, v in score.items():
        test_score[label + "_" + k] = v
    return test_score
