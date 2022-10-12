""" Scripts for losses """
from dataclasses import dataclass
import torch
import numpy as np
from torch import nn
from torch import Tensor


@dataclass
class LossConfig:
    """
    LossConfig class
    """
    lambda_bss: float
    lambda_gmgs: float
    score_mtx: torch.Tensor  # for GMGS


class Losser:
    """
    Losser class
    """
    def __init__(self, config: LossConfig, device: str = "cuda"):
        self.ce_loss = nn.CrossEntropyLoss().to(device)
        self.config = config
        self.accum = []

    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        Compute loss
        """
        loss = self.ce_loss(y_pred, torch.argmax(y_true, dim=1))
        gmgs_loss = self.calc_gmgs_loss(y_pred, y_true)
        bss_loss = self.calc_bss_loss(y_pred, y_true)
        loss = loss + \
            self.config.lambda_bss * bss_loss + \
            self.config.lambda_gmgs * gmgs_loss
        self.accum.append(loss.clone().detach().cpu().item())
        return loss

    def calc_gmgs_loss(self: Tensor, y_pred: Tensor, y_true) -> Tensor:
        """
        Compute GMGS loss
        """
        score_mtx = torch.tensor(self.config.score_mtx).cuda()
        y_truel = torch.argmax(y_true, dim=1)
        weight = score_mtx[y_truel]
        py = torch.log(y_pred)
        output = torch.mul(y_true, py)
        output = torch.mul(output, weight)
        output = torch.mean(output)
        return -output

    def calc_bss_loss(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        Compute BSS loss
        """
        tmp = y_pred - y_true
        tmp = torch.mul(tmp, tmp)
        tmp = torch.sum(tmp, dim=1)
        tmp = torch.mean(tmp)
        return tmp

    def get_mean_loss(self) -> float:
        """
        Get mean loss
        """
        return np.mean(self.accum)

    def clear(self):
        """
        Clear accumulated loss
        """
        self.accum.clear()
