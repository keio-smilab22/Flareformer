"""Train Flare Transformer"""

import json
import argparse

from torchinfo import summary

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import math
import torch.nn.functional as F

from src.model import FlareTransformerWithConvNext
from src.datasets import FlareDataset
from src.eval_utils import calc_score
from src.BalancedBatchSampler import TrainBalancedBatchSampler

import colored_traceback.always


def gmgs_loss_function(y_pred, y_true, score_matrix):
    """Compute GMGS loss"""
    score_matrix = torch.tensor(score_matrix).cuda()
    y_truel = torch.argmax(y_true, dim=1)
    weight = score_matrix[y_truel]
    py = torch.log(y_pred)
    output = torch.mul(y_true, py)
    output = torch.mul(output, weight)
    output = torch.mean(output)
    return -output


def label_smoothing(y_true, epsilon):
    """Return label smoothed vector"""
    x = y_true + epsilon
    x = x / (1+epsilon*4)
    return x


def bs_loss_function(y_pred, y_true):
    """Compute BSS loss"""
    tmp = y_pred - y_true
    tmp = torch.mul(tmp, tmp)
    tmp = torch.sum(tmp, dim=1)
    tmp = torch.mean(tmp)
    return tmp

def train_epoch(model, train_dl, epoch,lr,  args):
    """Return train loss and score for train set"""
    model.train()
    predictions = []
    observations = []
    train_loss = 0
    n = 0
    for _, (x, y, feat, idx) in enumerate(tqdm(train_dl)):
        if not args.without_schedule:
            adjust_learning_rate(optimizer, epoch, params["epochs"], lr, args)
        optimizer.zero_grad()
        output, feat = model(x.cuda().to(torch.float), feat.cuda().to(torch.float))

        # ib_loss = ib(output, torch.max(y, 1)[1].cuda().to(torch.long),feat)
        bce_loss = criterion(output, torch.max(y, 1)[1].cuda().to(torch.long))

        if params["lambda"]["GMGS"] != 0:
            gmgs_loss = gmgs_criterion(
                output, y.cuda().to(torch.float),
                args.dataset["GMGS_score_matrix"])
        else:
            gmgs_loss = 0

        if params["lambda"]["BS"] != 0:
            bs_loss = bs_criterion(output, y.cuda().to(torch.float))
        else:
            bs_loss = 0

        loss = bce_loss + \
            params["lambda"]["GMGS"] * gmgs_loss + \
            params["lambda"]["BS"] * bs_loss
        
        # if epoch < switch_epoch:
        #     loss = ib_loss + \
        #         params["lambda"]["GMGS"] * gmgs_loss + \
        #         params["lambda"]["BS"] * bs_loss
        # else:
        #     loss = ib_loss # ib_lossを使うように

        loss.backward()
        optimizer.step()

        train_loss += (loss.detach().cpu().item() * x.shape[0])
        n += x.shape[0]

        for pred, o in zip(output.cpu().detach().numpy().tolist(),
                           y.detach().numpy().tolist()):
            predictions.append(pred)
            observations.append(np.argmax(o))

    score = calc_score(predictions, observations,
                       args.dataset["climatology"])
    score = calc_test_score(score, "train")

    return score, train_loss/n


def eval_epoch(model, validation_dl):
    """Return val loss and score for val set"""
    model.eval()
    predictions = []
    observations = []
    valid_loss = 0
    n = 0
    with torch.no_grad():
        for _, (x, y, feat, idx) in enumerate(tqdm(validation_dl)):
            output, feat = model(x.cuda().to(torch.float),feat.cuda().to(torch.float))
                
            # bce_loss = criterion(output, y.cuda().to(torch.long))
            bce_loss = criterion(output, torch.max(y, 1)[1].cuda().to(torch.long))
            if params["lambda"]["GMGS"] != 0:
                gmgs_loss = gmgs_criterion(
                    output, y.cuda().to(torch.float),
                    args.dataset["GMGS_score_matrix"])
            else:
                gmgs_loss = 0
            if params["lambda"]["BS"] != 0:
                bs_loss = bs_criterion(output, y.cuda().to(torch.float))
            else:
                bs_loss = 0 
            loss = bce_loss + \
                params["lambda"]["GMGS"] * gmgs_loss + \
                params["lambda"]["BS"] * bs_loss
            valid_loss += (loss.detach().cpu().item() * x.shape[0])
            n += x.shape[0]
            for pred, o in zip(output.cpu().numpy().tolist(),
                               y.numpy().tolist()):
                predictions.append(pred)
                observations.append(np.argmax(o))
        score = calc_score(predictions, observations,
                           args.dataset["climatology"])
        score = calc_test_score(score, "valid")
    return score, valid_loss/n



def test_epoch(model, test_dl):
    """Return test loss and score for test set"""
    model.eval()
    predictions = []
    observations = []
    test_loss = 0
    n = 0
    with torch.no_grad():
        for _, (x, y, feat, idx) in enumerate(tqdm(test_dl)):
            output, feat = model(x.cuda().to(torch.float),feat.cuda().to(torch.float))

            # bce_loss = criterion(output, y.cuda().to(torch.long))
            bce_loss = criterion(output, torch.max(y, 1)[1].cuda().to(torch.long))
            if params["lambda"]["GMGS"] != 0:
                gmgs_loss = \
                    gmgs_criterion(output,
                                   y.cuda().to(torch.float),
                                   args.dataset["GMGS_score_matrix"])
            else:
                gmgs_loss = 0
            if params["lambda"]["BS"] != 0:
                bs_loss = bs_criterion(output, y.cuda().to(torch.float))
            else:
                bs_loss = 0
            loss = bce_loss +\
                params["lambda"]["GMGS"] * gmgs_loss +\
                params["lambda"]["BS"] * bs_loss
            test_loss += (loss.detach().cpu().item() * x.shape[0])
            n += x.shape[0]
            for pred, o in zip(output.cpu().numpy().tolist(), y.tolist()):
                predictions.append(pred)
                observations.append(np.argmax(o))
        score = calc_score(predictions, observations,
                           args.dataset["climatology"])
        score = calc_test_score(score, "test")
    return score, test_loss/n


def calc_test_score(score, label):
    """Return dict with key of label"""
    test_score = {}
    for k, v in score.items():
        test_score[label+"_"+k] = v
    return test_score

def adjust_learning_rate(optimizer, current_epoch, epochs, lr, args): # optimizerの内部パラメタを直接変えちゃうので注意
    """Decay the learning rate with half-cycle cosine after warmup"""
    min_lr = 0
    if current_epoch < args.warmup_epochs:
        lr = lr * epoch / args.warmup_epochs
    else:
        theta = math.pi * (current_epoch - args.warmup_epochs) / (epochs - args.warmup_epochs)
        lr = min_lr + (lr - min_lr) * 0.5 * (1. + math.cos(theta))

    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr

    return lr

def inject_args(args,target):
    for key, value in target.items():
        args.__setattr__(key, value)
    return args


if __name__ == "__main__":
    # fix seed value
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
 
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--params', default='params/params2017.json')
    parser.add_argument('--project_name', default='flare_transformer_test')
    parser.add_argument('--warmup_epochs', default=5, type=int)
    parser.add_argument('--without_schedule', action='store_false')
    parser.add_argument('--lr_stage2', default=0.000008, type=float)
    parser.add_argument('--epoch_stage2', default=25, type=float)
    parser.add_argument('--detail_summary', action='store_true')
    parser.add_argument('--imbalance', action='store_true')

    # read params/params.json
    args = parser.parse_args()
    params = json.loads(open(args.params).read())
    args = inject_args(args,params)

    # Initialize W&B
    if  args.wandb:
        wandb.init(project=args.project_name, name=params["wandb_name"])

    print("==========================================")
    print(json.dumps(params, indent=2))
    print("==========================================")

    # Initialize Dataset
    train_dataset = FlareDataset("train", args.dataset)
    validation_dataset = FlareDataset("valid", args.dataset)
    test_dataset = FlareDataset("test", args.dataset)

    mean, std = train_dataset.calc_mean()
    
    train_dataset.set_mean(mean, std)    
    validation_dataset.set_mean(mean, std)
    test_dataset.set_mean(mean, std)
    print(mean, std)
    
    print("Batch Sampling")

    if args.imbalance:
        train_dl = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=2)
    else:
        train_dl = DataLoader(train_dataset, batch_sampler=TrainBalancedBatchSampler(
            train_dataset, args.output_channel, args.bs//args.output_channel))

    validation_dl = DataLoader(validation_dataset, batch_size=args.bs, shuffle=False,num_workers=2)
    test_dl = DataLoader(test_dataset, batch_size=args.bs, shuffle=False,num_workers=2)

    # Initialize Loss Function
    criterion = nn.CrossEntropyLoss().cuda()
    gmgs_criterion = gmgs_loss_function
    bs_criterion = bs_loss_function

    model = FlareTransformerWithConvNext(input_channel=args.input_channel,
                                        output_channel=args.output_channel,
                                        sfm_params=args.SFM,
                                        mm_params=args.MM,
                                        window=args.dataset["window"]).to("cuda")
    
    if args.detail_summary:
        summary(model,[(args.bs, *sample.shape) for sample in train_dataset[0][0::2]])
    else:
        summary(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Start Training
    best_score = {}
    best_score["valid_"+params["main_metric"]] = -10
    best_epoch = 0
    model_update_dict = {}
    for e, epoch in enumerate(range(params["epochs"])):
        print("====== Epoch ", e, " ======")
        train_score, train_loss = train_epoch(model, train_dl, epoch, args.lr, args)
        valid_score, valid_loss = eval_epoch(model, validation_dl)
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

        if  args.wandb is True:
            wandb.log(log)

        print("Epoch {}: Train loss:{:.4f}  Valid loss:{:.4f}".format(e, train_loss, valid_loss), test_score)


    # Output Test Score
    print("========== TEST ===========")
    model.load_state_dict(torch.load(params["save_model_path"])) 
    test_score, _ = test_epoch(model, test_dl)
    print("epoch : ", best_epoch, test_score)
    if  args.wandb is True:
        wandb.log(calc_test_score(test_score, "final"))

    # ここからCRT
    if args.imbalance:
        print("Start CRT")
        train_dl = DataLoader(train_dataset, batch_sampler=TrainBalancedBatchSampler(
            train_dataset, args.output_channel, args.bs//args.output_channel))
        
        model.freeze_feature_extractor()
        summary(model)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_stage2)
        for e, epoch in enumerate(range(args.epoch_stage2)):
            print("====== Epoch ", e, " ======")
            train_score, train_loss = train_epoch(model, train_dl, epoch, args.lr_stage2, args)
            valid_score, valid_loss = eval_epoch(model, validation_dl)
            test_score, test_loss = valid_score, valid_loss

            log = {'epoch': epoch, 'train_loss': np.mean(train_loss),
                'valid_loss': np.mean(valid_loss),
                'test_loss': np.mean(test_loss)}
            log.update(train_score)
            log.update(valid_score)
            log.update(test_score)

            if  args.wandb is True:
                wandb.log(log)

            print("Epoch {}: Train loss:{:.4f}  Valid loss:{:.4f}".format(
                e, train_loss, valid_loss), test_score)
