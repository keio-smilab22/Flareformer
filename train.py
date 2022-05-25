"""Train Flare Transformer"""

import json
import argparse

from torchinfo import summary

import numpy as np
import torch
from torch import nn
from utils.utils import *
from utils.losses import LossConfig, Losser
from engine import calc_test_score, train_epoch, eval_epoch
import wandb

from models.model import FlareFormer
from datasets.datasets import prepare_dataloaders

import colored_traceback.always


def parse_params():
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
    return args, params


if __name__ == "__main__":
    fix_seed(seed=42)
    args, params = parse_params()

    # Initialize W&B
    if args.wandb:
        wandb.init(project=args.project_name, name=args.wandb_name)

    print("==========================================")
    print(json.dumps(params, indent=2))
    print("==========================================")

    # Initialize Dataset

    print("Prepare Dataloaders")
    (train_dl, val_dl, test_dl), sample = prepare_dataloaders(args, args.imbalance)

    # Initialize Loss Function
    loss_config = LossConfig(lambda_bss=args.factor["BS"],
                             lambda_gmgs=args.factor["GMGS"],
                             score_mtx=args.dataset["GMGS_score_matrix"])

    losser = Losser(loss_config)

    model = FlareFormer(input_channel=args.input_channel,
                        output_channel=args.output_channel,
                        sfm_params=args.SFM,
                        mm_params=args.MM,
                        window=args.dataset["window"]).to("cuda")

    if args.detail_summary:
        summary(model, [(args.bs, *feature.shape) for feature in sample[0::2]])
    else:
        summary(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Start Training
    best_score = {}
    best_score["valid_" + params["main_metric"]] = -10
    best_epoch = 0
    model_update_dict = {}
    for e, epoch in enumerate(range(params["epochs"])):
        print("====== Epoch ", e, " ======")
        train_score, train_loss = train_epoch(model, optimizer, train_dl, epoch, args.lr, args, losser)
        valid_score, valid_loss = eval_epoch(model, val_dl, losser, args)
        test_score, test_loss = valid_score, valid_loss

        torch.save(model.state_dict(), params["save_model_path"])
        best_score = valid_score
        best_epoch = e

        log = {'epoch': epoch, 'train_loss': np.mean(train_loss),
               'valid_loss': np.mean(valid_loss),
               'test_loss': np.mean(test_loss)}
        log.update(train_score)
        log.update(valid_score)
        log.update(test_score)

        if args.wandb is True:
            wandb.log(log)

        print("Epoch {}: Train loss:{:.4f}  Valid loss:{:.4f}".format(e, train_loss, valid_loss), test_score)

    # Output Test Score
    print("========== TEST ===========")
    model.load_state_dict(torch.load(params["save_model_path"]))
    test_score, _ = eval_epoch(model, test_dl, losser, args)
    print("epoch : ", best_epoch, test_score)
    if args.wandb is True:
        wandb.log(calc_test_score(test_score, "final"))

    # ここからCRT
    if args.imbalance:
        print("Start CRT")
        (train_dl, val_dl, test_dl), sample = prepare_dataloaders(args, not args.imbalance)

        model.freeze_feature_extractor()
        summary(model)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_for_stage2)
        for e, epoch in enumerate(range(args.epoch_for_2stage)):
            print("====== Epoch ", e, " ======")
            train_score, train_loss = train_epoch(model, optimizer, train_dl, epoch, args.lr_for_stage2, args, losser)
            valid_score, valid_loss = eval_epoch(model, val_dl, losser, args)
            test_score, test_loss = valid_score, valid_loss

            log = {'epoch': epoch, 'train_loss': np.mean(train_loss),
                   'valid_loss': np.mean(valid_loss),
                   'test_loss': np.mean(test_loss)}
            log.update(train_score)
            log.update(valid_score)
            log.update(test_score)

            if args.wandb is True:
                wandb.log(log)

            print("Epoch {}: Train loss:{:.4f}  Valid loss:{:.4f}".format(
                e, train_loss, valid_loss), test_score)
