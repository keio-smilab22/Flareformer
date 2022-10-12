""" Scripts for logs """
from argparse import Namespace
from dataclasses import dataclass
from typing import Any, Dict, List
import wandb as wandb_runner


@dataclass
class Log:
    """ Log class """
    stage: str
    loss: float
    score: Any


class Logger:
    """ Logger class """
    def __init__(self, args: Namespace, wandb: bool) -> None:
        if args.wandb:
            wandb_runner.init(project=args.project_name, name=args.model_name)

        self.wandb_enabled = wandb

    def write(self, epoch: int, logs: List[Log]):
        """ Write logs """
        _l: Dict[str, Any] = {"epoch": epoch}
        for _lg in logs:
            _l[f"{_lg.stage}_loss"] = _lg.loss
            _l.update(_lg.score)

        if self.wandb_enabled:
            wandb_runner.log(_l)
