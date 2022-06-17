"""Train Flare Transformer"""

import json
import argparse
import torch
import models.model
import torch.nn as nn
import numpy as np
import colored_traceback.always

from argparse import Namespace
from typing import Dict, Tuple, Any
from torchinfo import summary
from utils.statistics import Stat

from utils.utils import adjust_learning_rate, fix_seed, inject_args
from utils.losses import LossConfig, Losser
from engine import train_epoch, eval_epoch
from utils.logs import Log, Logger

from datasets.datasets import prepare_dataloaders


def parse_params(dump: bool = False) -> Tuple[Namespace, Dict[str, Any]]:
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--model', default='Flareformer')
    parser.add_argument('--params', default='params/params_2017.json')
    parser.add_argument('--project_name', default='flare_transformer_test')
    parser.add_argument('--model_name', default='id1_2017')
    parser.add_argument('--warmup_epochs', default=5, type=int)
    parser.add_argument('--without_schedule', action='store_false')
    parser.add_argument('--lr_for_2stage', default=0.000008, type=float)
    parser.add_argument('--epoch_for_2stage', default=25, type=int)
    parser.add_argument('--detail_summary', action='store_true')
    parser.add_argument('--imbalance', action='store_true')
    parser.add_argument('--debug', action='store_true')

    # read params/params.json
    args = parser.parse_args()
    params = json.loads(open(args.params).read())
    args = inject_args(args, params)

    if dump:
        print("==========================================")
        print(json.dumps(params, indent=2))
        print("==========================================")

    return args, params


class FlareformerManager():
    """
    Manager class for Flareformer
    """

    def __init__(self, args: Namespace):
        # init seed and logger
        fix_seed(seed=42)
        logger = Logger(args, wandb=args.wandb)

        # Prepare dataloaders
        dataloaders, sample = prepare_dataloaders(args, args.debug, args.imbalance)

        # Prepare model and optimizer
        model, losser, optimizer, stat = self._build(args, sample)

        self.dataloaders = dataloaders  # (train, valid, test)
        self.model = model
        self.logger = logger
        self.losser = losser
        self.optimizer = optimizer
        self.stat = stat
        self.args = args
        self.mock_sample = sample

    def train(self, lr=None, epochs=None):
        """
        Train model
        """
        lr = lr or self.args.lr
        (train_dl, val_dl, _) = self.dataloaders
        for epoch in range(epochs or self.args.epochs):
            print(f"====== Epoch {epoch} ======")

            # learning rate scheduler
            if not self.args.without_schedule:
                adjust_learning_rate(self.optimizer,
                                     epoch,
                                     self.args.dataset["epochs"],
                                     lr,
                                     self.args)

            # train
            train_score, train_loss = train_epoch(self.model,
                                                  self.optimizer,
                                                  train_dl,
                                                  losser=self.losser,
                                                  stat=self.stat)

            # validation
            valid_score, valid_loss = eval_epoch(self.model,
                                                 val_dl,
                                                 losser=self.losser,
                                                 stat=self.stat)

            # test
            test_score, test_loss = valid_score, valid_loss

            # log & save
            torch.save(self.model.state_dict(), self.args.save_model_path)
            self.logger.write(epoch, [Log("train", np.mean(train_loss), train_score),
                                      Log("valid", np.mean(valid_loss), valid_score),
                                      Log("test", np.mean(test_loss), test_score)])

            print("Epoch {}: Train loss:{:.4f}  Valid loss:{:.4f}".format(epoch, train_loss, valid_loss), test_score)

    def load(self, path: str):
        """
        Load model from path
        """
        self.model.load_state_dict(torch.load(path))

    def test(self):
        """
        Test model
        """
        (_, _, test_dl) = self.dataloaders
        test_score, _ = eval_epoch(self.model,
                                   test_dl,
                                   losser=self.losser,
                                   stat=self.stat)
        print(test_score)

    def print_summary(self):
        """
        Print model summary
        """
        if self.args.detail_summary:
            summary(self.model, [(self.args.bs, *feature.shape) for feature in self.mock_sample[0]])
        else:
            summary(self.model)

    def freeze_feature_extractor(self):
        """
        Freeze feature extractor
        """
        self.model.freeze_feature_extractor()

    def reset_optimizer(self, lr=None):
        """
        Reset optimizer with new lr
        """
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr or self.args.lr)

    def _build(self, args: Namespace, sample: Any) -> Tuple[nn.Module, Losser, torch.optim.Adam, Stat]:
        """
        Build model, losser, optimizer, stat
        """
        print("Prepare model and optimizer", end="")
        loss_config = LossConfig(lambda_bss=args.factor["BS"],
                                 lambda_gmgs=args.factor["GMGS"],
                                 score_mtx=args.dataset["GMGS_score_matrix"])

        # Model
        Model = self._get_model_class(args.model)
        model = Model(input_channel=args.input_channel,
                      output_channel=args.output_channel,
                      sfm_params=args.SFM,
                      mm_params=args.MM,
                      window=args.dataset["window"]).to("cuda")

        # Optimizer & Stat & Loss Function
        losser = Losser(loss_config)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        stat = Stat(args.dataset["climatology"])

        print(" ... ok")
        if args.detail_summary:
            summary(model, [(args.bs, *feature.shape) for feature in sample[0]])
        else:
            summary(model)

        return model, losser, optimizer, stat

    def _get_model_class(self, name: str) -> nn.Module:
        mclass = models.model.__dict__[name]
        return mclass


def main():
    args, _ = parse_params(dump=True)
    flareformer = FlareformerManager(args)

    print("Start training\n")
    flareformer.train()

    print("\n========== eval ===========")
    flareformer.load(args.save_model_path)
    flareformer.test()

    if args.imbalance:
        print("Start cRT (Classifier Re-training)")
        flareformer.freeze_feature_extractor()
        flareformer.reset_optimizer(lr=args.lr_for_2stage)
        flareformer.print_summary()
        flareformer.train(lr=args.lr_for_2stage,
                          epochs=args.epoch_for_2stage)


if __name__ == "__main__":
    main()
