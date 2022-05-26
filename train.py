"""Train Flare Transformer"""

import json
import argparse
import torch
import torch.nn as nn
import numpy as np
import colored_traceback.always

from argparse import Namespace
from typing import Dict, Tuple, Any
from torchinfo import summary

from utils.utils import fix_seed, inject_args
from utils.losses import LossConfig, Losser
from engine import train_epoch, eval_epoch
from utils.logs import Log, Logger

from models.model import FlareFormer
from datasets.datasets import prepare_dataloaders


def parse_params(dump: bool = False) -> Tuple[Namespace, Dict[str, Any]]:
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--params', default='params/params2017.json')
    parser.add_argument('--project_name', default='flare_transformer_test')
    parser.add_argument('--warmup_epochs', default=5, type=int)
    parser.add_argument('--without_schedule', action='store_false')
    parser.add_argument('--lr_for_stage2', default=0.000008, type=float)
    parser.add_argument('--epoch_for_2stage', default=25, type=int)
    parser.add_argument('--detail_summary', action='store_true')
    parser.add_argument('--imbalance', action='store_true')

    # read params/params.json
    args = parser.parse_args()
    params = json.loads(open(args.params).read())
    args = inject_args(args, params)

    if dump:
        print("==========================================")
        print(json.dumps(params, indent=2))
        print("==========================================")

    return args, params


def build(args: Namespace, sample: Any) -> Tuple[nn.Module, Losser, torch.optim.Adam]:
    # Model
    model = FlareFormer(input_channel=args.input_channel,
                        output_channel=args.output_channel,
                        sfm_params=args.SFM,
                        mm_params=args.MM,
                        window=args.dataset["window"]).to("cuda")

    if args.detail_summary:
        summary(model, [(args.bs, *feature.shape) for feature in sample[0::2]])
    else:
        summary(model)

    # Loss Function
    loss_config = LossConfig(lambda_bss=args.factor["BS"],
                             lambda_gmgs=args.factor["GMGS"],
                             score_mtx=args.dataset["GMGS_score_matrix"])
    losser = Losser(loss_config)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    return model, losser, optimizer


def main() -> None:
    # init
    fix_seed(seed=42)
    args, params = parse_params(dump=True)
    logger = Logger(args, wandb=args.wandb)

    # Initialize Dataset
    print("Prepare Dataloaders")
    (train_dl, val_dl, test_dl), sample = prepare_dataloaders(args, args.imbalance)

    print("Prepare model and optimizer")
    model, losser, optimizer = build(args, sample)

    # Training
    print("Start training\n")
    for epoch in range(args.epochs):
        print(f"====== Epoch {epoch} ======")
        train_score, train_loss = train_epoch(model, optimizer, train_dl, epoch, args.lr, args, losser)
        valid_score, valid_loss = eval_epoch(model, val_dl, losser, args)
        test_score, test_loss = valid_score, valid_loss

        torch.save(model.state_dict(), params["save_model_path"])
        logger.write(epoch, [Log("train", np.mean(train_loss), train_score),
                             Log("valid", np.mean(valid_loss), valid_score),
                             Log("test", np.mean(test_loss), test_score)])

        print("Epoch {}: Train loss:{:.4f}  Valid loss:{:.4f}".format(epoch, train_loss, valid_loss), test_score)

    # Evaluate
    print("\n========== eval ===========")
    model.load_state_dict(torch.load(params["save_model_path"]))
    test_score, _ = eval_epoch(model, test_dl, losser, args)
    print(test_score)

    # cRT
    print("Start cRT(Classifier Re-training)")
    if args.imbalance:
        (train_dl, val_dl, test_dl), sample = prepare_dataloaders(args, not args.imbalance)

        model.freeze_feature_extractor()
        summary(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_for_stage2)

        for epoch in range(args.epoch_for_2stage):
            print(f"====== Epoch {epoch} ======")
            train_score, train_loss = train_epoch(model, optimizer, train_dl, epoch, args.lr_for_stage2, args, losser)
            valid_score, valid_loss = eval_epoch(model, val_dl, losser, args)
            test_score, test_loss = valid_score, valid_loss

            logger.write(epoch, [Log("train", np.mean(train_loss), train_score),
                                 Log("valid", np.mean(valid_loss), valid_score),
                                 Log("test", np.mean(test_loss), test_score)])

            print("Epoch {}: Train loss:{:.4f}  Valid loss:{:.4f}".format(epoch, train_loss, valid_loss), test_score)


if __name__ == "__main__":
    main()
