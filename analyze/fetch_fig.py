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
from datasets.flare import FlareDataset
from models.model import FlareFormer

from utils.utils import fix_seed, inject_args
from utils.losses import LossConfig, Losser
from engine import train_epoch, eval_epoch
from utils.logs import Log, Logger

# from models.model import FlareFormer
import models
from datasets.datasets import prepare_dataloaders
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm


def parse_params(dump: bool = False) -> Tuple[Namespace, Dict[str, Any]]:
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--model', default='FlareFormer')
    parser.add_argument('--params', default='params/params_2017.json')
    parser.add_argument('--project_name', default='flare_transformer_test')
    parser.add_argument('--model_name', default='id1_2017')
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


def get_model_class(name: str) -> nn.Module:
    mclass = models.model.__dict__[name]
    return mclass


def build(args: Namespace, sample: Any) -> Tuple[nn.Module, Losser, torch.optim.Adam]:
    # Model
    Model = get_model_class(args.model)
    model = Model(input_channel=args.input_channel,
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


def read_meta_jsonl():
    with open("images/ft_database_all17.jsonl", "r") as f:
        return [json.loads(line) for line in f]


def main() -> None:
    # init
    fix_seed(seed=42)
    args, params = parse_params(dump=True)
    logger = Logger(args, wandb=args.wandb)

    # Initialize Dataset
    print("Prepare Dataloaders")
    train_dataset = FlareDataset("train", args.dataset)
    test_dataset = FlareDataset("test", args.dataset)
    (train_dl, val_dl, test_dl), sample = prepare_dataloaders(args, args.imbalance)

    print("Prepare model and optimizer")
    model, losser, optimizer = build(args, sample)
    model.load_state_dict(torch.load("checkpoints/id23_2017_well_second.pth"))

    # Prepare sample images
    model.eval()
    meta = read_meta_jsonl()
    with torch.no_grad():
        for _, (x, y, idx) in enumerate(tqdm(val_dl)):
            imgs, feats = x
            imgs, feats = imgs.cuda().float(), feats.cuda().float()
            output, _ = model(imgs, feats)
            gt = y.cuda().to(torch.float)
            s = "OCMX"
            st = [(3, 3), (2, 2), (2, 1)]
            seen = [False for _ in range(3)]
            for i, (pred, o) in enumerate(zip(output.cpu().numpy().tolist(), y.numpy().tolist())):
                for l, (p, gt) in enumerate(st):
                    if seen[l]:
                        continue
                    if np.argmax(pred) == p and np.argmax(o) == gt:
                        import cv2 as cv
                        img, _ = x
                        img = img[i]
                        K, C, H, W = img.shape
                        image = []
                        mean, std = 0.36368870735168457, 0.22177115364615466
                        for k in range(K):
                            col_img = np.empty((H, W, 3))
                            for j in range(3):
                                col_img[:, :, j] = (img[k, 0, :, :] * std + mean) * 255

                            window_idx = test_dataset.window[idx[i]][:K] + len(train_dataset)
                            timestamp = meta[window_idx[k]]["time"]
                            cv.imwrite(f"images/ID={window_idx[k]}_p={s[p]}_gt={s[gt]}_({timestamp}).png", col_img)
                        print(f"saved. (p={s[p]}_gt={s[gt]})_{timestamp}_{pred}")
                        seen[l] = True

    # # cRT
    # print("Start cRT(Classifier Re-training)")
    # if args.imbalance:
    #     (train_dl, val_dl, test_dl), sample = prepare_dataloaders(args, not args.imbalance)

    #     model.freeze_feature_extractor()
    #     summary(model)
    #     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_for_stage2)

    #     for epoch in range(args.epoch_for_2stage):
    #         print(f"====== Epoch {epoch} ======")
    #         train_score, train_loss = train_epoch(model, optimizer, train_dl, epoch, args.lr_for_stage2, args, losser)
    #         valid_score, valid_loss = eval_epoch(model, val_dl, losser, args)
    #         test_score, test_loss = valid_score, valid_loss

    #         logger.write(epoch, [Log("train", np.mean(train_loss), train_score),
    #                              Log("valid", np.mean(valid_loss), valid_score),
    #                              Log("test", np.mean(test_loss), test_score)])

    #         if epoch == 2:
    #             torch.save(model.state_dict(), "checkpoints/id23_2017_well_second.pth")

    #         print("Epoch {}: Train loss:{:.4f}  Valid loss:{:.4f}".format(epoch, train_loss, valid_loss), test_score)


if __name__ == "__main__":
    main()
