from dataclasses import dataclass
from torch import nn

import torch


@dataclass
class LossConfig:
    lambda_bss: float
    lambda_gmgs: float
    score_mtx: torch.Tensor

class Losser:
    def __init__(self,config: LossConfig, device: str = "cuda"):
        self.ce_loss = nn.CrossEntropyLoss().to(device)
        self.config = config

    def __call__(self, y_pred, y_true):
        loss = self.ce_loss(y_pred,torch.argmax(y_true,dim=1))
        gmgs_loss = self.calc_gmgs_loss(y_pred,y_true)
        bss_loss = self.calc_bss_loss(y_pred,y_true)
        return  loss + \
                self.config.lambda_bss * bss_loss + \
                self.config.lambda_gmgs * gmgs_loss
        
    def calc_gmgs_loss(self,y_pred, y_true):
        """Compute GMGS loss"""
        score_mtx = torch.tensor(self.config.score_mtx).cuda()
        y_truel = torch.argmax(y_true, dim=1)
        weight = score_mtx[y_truel]
        py = torch.log(y_pred)
        output = torch.mul(y_true, py)
        output = torch.mul(output, weight)
        output = torch.mean(output)
        return -output

    def calc_bss_loss(self,y_pred, y_true):
        """Compute BSS loss"""
        tmp = y_pred - y_true
        tmp = torch.mul(tmp, tmp)
        tmp = torch.sum(tmp, dim=1)
        tmp = torch.mean(tmp)
        return tmp
