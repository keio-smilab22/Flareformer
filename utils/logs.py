import wandb as wandb_runner
from argparse import Namespace
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class Log:
    stage: str
    loss: float
    score: Any


class Logger:
    def __init__(self, args: Namespace, wandb: bool) -> None:
        if args.wandb:
            wandb_runner.init(project=args.project_name, name=args.model_name)

        self.wandb_enabled = wandb

    def write(self, epoch: int, logs: List[Log]):
        l: Dict[str, Any] = {"epoch": epoch}
        for lg in logs:
            l[f"{lg.stage}_loss"] = lg.loss
            l.update(lg.score)

        if self.wandb_enabled:
            wandb_runner.log(l)
