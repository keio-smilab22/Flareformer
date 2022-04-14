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

from src.model import FlareTransformer, _FlareTransformerWithGAPMAE, FlareTransformerLikeViLBERT, FlareTransformerReplacedFreezeViTWithMAE, FlareTransformerReplacedViTWithMAE, FlareTransformerWith1dMAE, FlareTransformerWithConvNext, FlareTransformerWithGAPMAE, FlareTransformerWithGAPSeqMAE, FlareTransformerWithMAE, FlareTransformerWithPositonalEncoding, FlareTransformerWithoutMM, FlareTransformerWithoutPE, PureTransformerSFM
from src.Dataloader import CombineDataloader, TrainDataloader, TrainDataloader256
from src.eval_utils import calc_score
from src.BalancedBatchSampler import TrainBalancedBatchSampler

import colored_traceback.always

def gmgs_loss_function(y_pred, y_true, score_matrix):
    """Compute GMGS loss"""
    score_matrix = torch.tensor(score_matrix).cuda()
    y_truel = torch.argmax(y_true, dim=1)
    weight = score_matrix[y_truel]
    py = torch.log(y_pred)
    # y_true = label_smoothing(y_true, 0.01)
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


def train_epoch(model, train_dl):
    """Return train loss and score for train set"""
    model.train()
    predictions = []
    observations = []
    train_loss = 0
    n = 0
    for _, (x, y, feat) in enumerate(tqdm(train_dl)):
        optimizer.zero_grad()
        output = model(x.cuda().to(torch.float), feat.cuda().to(torch.float))
        bce_loss = criterion(output, torch.max(y, 1)[
                             1].cuda().to(torch.long))

        if params["lambda"]["GMGS"] != 0:
            gmgs_loss = gmgs_criterion(
                output, y.cuda().to(torch.float),
                params["dataset"]["GMGS_score_matrix"])
        else:
            gmgs_loss = 0

        if params["lambda"]["BS"] != 0:
            bs_loss = bs_criterion(output, y.cuda().to(torch.float))
        else:
            bs_loss = 0
        loss = bce_loss + \
            params["lambda"]["GMGS"] * gmgs_loss + \
            params["lambda"]["BS"] * bs_loss
        loss.backward()
        optimizer.step()

        train_loss += (loss.detach().cpu().item() * x.shape[0])
        n += x.shape[0]

        for pred, o in zip(output.cpu().detach().numpy().tolist(),
                           y.detach().numpy().tolist()):
            predictions.append(pred)
            observations.append(np.argmax(o))

    score = calc_score(predictions, observations,
                       params["dataset"]["climatology"])
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
        for _, (x, y, feat) in enumerate(tqdm(validation_dl)):
            output = model(x.cuda().to(torch.float),
                           feat.cuda().to(torch.float))
            bce_loss = criterion(output, torch.max(y, 1)[
                                 1].cuda().to(torch.long))
            if params["lambda"]["GMGS"] != 0:
                gmgs_loss = gmgs_criterion(
                    output, y.cuda().to(torch.float),
                    params["dataset"]["GMGS_score_matrix"])
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
                           params["dataset"]["climatology"])
        score = calc_test_score(score, "valid")
    return score, valid_loss/n




def calc_model_update(model, dl,model_update_dict): # モデルの更新量を計算する
    model.eval()
    with torch.no_grad():
        for _, (x, y, feat) in enumerate(tqdm(dl)):
            x = x.cuda().to(torch.float)
            feat = feat.cuda().to(torch.float)
            mm_extractor = model.forward_mm_feature_extractor
            sfm_extractor = model.forward_sfm_feature_extractor
            mm_output = mm_extractor(x,feat).cpu().numpy() # (B, W, d_model)
            sfm_output = sfm_extractor(x,feat).cpu().numpy()
            # mm_mean = mm_output.cpu().numpy().mean(axis=0)
            # sfm_mean = sfm_output.cpu().numpy().mean(axis=0)

            # print(mm_output.shape)
            # print(sfm_output.shape)
            
            if "mm" not in model_update_dict: model_update_dict["mm"] = [mm_output]
            else:
                z = model_update_dict["mm"][0]
                z = z - mm_output
                t = [np.linalg.norm(z[i,0,:]) for i in range(z.shape[0])] # ウィンドウのheadだけ
                model_update_dict["mm"].append(np.mean(t))

            if "sfm" not in model_update_dict: model_update_dict["sfm"] = [sfm_output]
            else:
                z = model_update_dict["sfm"][0]
                z = z - sfm_output
                t = [np.linalg.norm(z[i,0,:]) for i in range(z.shape[0])] # ウィンドウのheadだけ
                model_update_dict["sfm"].append(np.mean(t))

            break
        # for pred, o in zip(output.cpu().numpy().tolist(),
        #                     y.numpy().tolist()):
        #     predictions.append(pred)
        #     observations.append(np.argmax(o))

    print("+-- model update --+")
    for k,v in model_update_dict.items():
        print(f" {k} = ", end='')
        print(v[1:])

    print("+------------------+\n")


def test_epoch(model, test_dl):
    """Return test loss and score for test set"""
    model.eval()
    predictions = []
    observations = []
    test_loss = 0
    n = 0
    with torch.no_grad():
        for _, (x, y, feat) in enumerate(tqdm(test_dl)):
            output = model(x.cuda().to(torch.float),
                           feat.cuda().to(torch.float))
            bce_loss = criterion(output, torch.max(y, 1)[
                                 1].cuda().to(torch.long))
            if params["lambda"]["GMGS"] != 0:
                gmgs_loss = \
                    gmgs_criterion(output,
                                   y.cuda().to(torch.float),
                                   params["dataset"]["GMGS_score_matrix"])
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
                           params["dataset"]["climatology"])
        score = calc_test_score(score, "test")
    return score, test_loss/n


def calc_test_score(score, label):
    """Return dict with key of label"""
    test_score = {}
    for k, v in score.items():
        test_score[label+"_"+k] = v
    return test_score


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
    parser.add_argument('--baseline', default='attn')
    parser.add_argument('--has_vit_head', action='store_true')
    parser.add_argument('--dim', default=512, type=int)
    parser.add_argument('--enc_depth', default=12, type=int)
    parser.add_argument('--dec_depth', default=8, type=int)
    parser.add_argument('--token_window', default=4, type=int)
    
    args = parser.parse_args()
    wandb_flag = args.wandb

    # read params/params.json
    params = json.loads(open(args.params).read())

    # Initialize W&B
    if wandb_flag is True:
        wandb.init(project=args.project_name, name=params["wandb_name"])

    print("==========================================")
    print(json.dumps(params, indent=2))
    print("==========================================")

    # Initialize Dataset
    # train_dataset = TrainDataloader("train", params["dataset"])
    train_dataset = TrainDataloader256("train", params["dataset"])
    # aug_train_dataset = TrainDataloader("train", params["dataset"], augmentation=True)
    if params["dataset"]["mean"] != 0:
        mean = params["dataset"]["mean"]
        std = params["dataset"]["std"]
    else:
        mean, std = train_dataset.calc_mean()
        print(mean, std)

    mean, std = train_dataset.calc_mean()
    print(mean, std)
    train_dataset.set_mean(mean, std)
    
    validation_dataset = TrainDataloader256("valid", params["dataset"])
    validation_dataset.set_mean(mean, std)
    test_dataset = TrainDataloader256("test", params["dataset"])
    test_dataset.set_mean(mean, std)

    print("Batch Sampling")
    # train_dl = DataLoader(train_dataset, batch_size=params["bs"], shuffle=True)
    train_dl = DataLoader(train_dataset, batch_sampler=TrainBalancedBatchSampler(
        train_dataset, params["output_channel"], params["bs"]//params["output_channel"]))
    validation_dl = DataLoader(validation_dataset,
                               batch_size=params["bs"], shuffle=False)
    test_dl = DataLoader(test_dataset, batch_size=params["bs"], shuffle=False)

    # Initialize Loss Function
    criterion = nn.CrossEntropyLoss().cuda()
    gmgs_criterion = gmgs_loss_function
    bs_criterion = bs_loss_function

    # model = FlareTransformerWithMAE(input_channel=params["input_channel"],
    #                          output_channel=params["output_channel"],
    #                          sfm_params=params["SFM"],
    #                          mm_params=params["MM"],
    #                          window=params["dataset"]["window"],
    #                          baseline=args.baseline,
    #                          embed_dim = args.dim,
    #                          enc_depth=args.enc_depth,
    #                          dec_depth=args.dec_depth,
    #                          has_vit_head=args.has_vit_head).to("cuda")


    # model = FlareTransformerWith1dMAE(input_channel=params["input_channel"],
    #                          output_channel=params["output_channel"],
    #                          sfm_params=params["SFM"],
    #                          mm_params=params["MM"],
    #                          window=params["dataset"]["window"],
    #                          token_window=args.token_window).to("cuda")


    # model = _FlareTransformerWithGAPMAE(input_channel=params["input_channel"],
    #                          output_channel=params["output_channel"],
    #                          sfm_params=params["SFM"],
    #                          mm_params=params["MM"],
    #                          window=params["dataset"]["window"],
    #                          baseline=args.baseline,
    #                          embed_dim = args.dim,
    #                          enc_depth=args.enc_depth,
    #                          dec_depth=args.dec_depth).to("cuda")


    model = FlareTransformer(input_channel=params["input_channel"],
                             output_channel=params["output_channel"],
                             sfm_params=params["SFM"],
                             mm_params=params["MM"],
                             window=params["dataset"]["window"]).to("cuda")

    # model = FlareTransformerWithGAPSeqMAE(input_channel=params["input_channel"],
    #                                     output_channel=params["output_channel"],
    #                                     sfm_params=params["SFM"],
    #                                     mm_params=params["MM"],
    #                                     window=params["dataset"]["window"],
    #                                     baseline=args.baseline,
    #                                     embed_dim = args.dim,
    #                                     enc_depth=args.enc_depth,
    #                                     dec_depth=args.dec_depth,
    #                                     need_cnn=False).to("cuda")


    summary(model,[(params["bs"], *train_dataset[0][0].shape),(params["bs"], *train_dataset[0][2].shape)])
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])

    # Start Training
    best_score = {}
    best_score["valid_"+params["main_metric"]] = -10
    best_epoch = 0
    model_update_dict = {}
    for e, epoch in enumerate(range(params["epochs"])):
        print("====== Epoch ", e, " ======")
        train_score, train_loss = train_epoch(model, train_dl)
        valid_score, valid_loss = eval_epoch(model, validation_dl)
        # test_score, test_loss = test_epoch(model, test_dl) # for train/val/test model
        test_score, test_loss = valid_score, valid_loss

        if best_score["valid_"+params["main_metric"]] < \
                valid_score["valid_"+params["main_metric"]]:
            torch.save(model.state_dict(), params["save_model_path"])
            best_score = valid_score
            best_epoch = e

        log = {'epoch': epoch, 'train_loss': np.mean(train_loss),
               'valid_loss': np.mean(valid_loss),
               'test_loss': np.mean(test_loss)}
        log.update(train_score)
        log.update(valid_score)
        log.update(test_score)

        if wandb_flag is True:
            wandb.log(log)

        print("Epoch {}: Train loss:{:.4f}  Valid loss:{:.4f}".format(
              e, train_loss, valid_loss), test_score)

        # calc_model_update(model,train_dl,model_update_dict)

    # Output Test Score
    print("========== TEST ===========")
    model.load_state_dict(torch.load(params["save_model_path"])) 
    test_score, _ = test_epoch(model, test_dl)
    print("epoch : ", best_epoch, test_score)
    if wandb_flag is True:
        wandb.log(calc_test_score(test_score, "final"))
