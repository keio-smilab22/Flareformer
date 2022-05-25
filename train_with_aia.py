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

from src.model import FlareTransformer, _FlareTransformerWithGAPMAE, FlareTransformerAIA, FlareTransformerLikeViLBERT, FlareTransformerReplacedFreezeViTWithMAE, FlareTransformerReplacedViTWithMAE, FlareTransformerWith1dMAE, FlareTransformerWithConvNext, FlareTransformerWithConvNextAIA, FlareTransformerWithGAPMAE, FlareTransformerWithGAPSeqMAE, FlareTransformerWithMAE, FlareTransformerWithMultiPE, FlareTransformerWithPE, FlareTransformerWithPositonalEncoding, FlareTransformerWithoutMM, FlareTransformerWithoutPE, PureTransformerSFM
from src.Dataloader import CombineDataloader, TrainDataloader, TrainDataloader256
from src.eval_utils import calc_score
from src.BalancedBatchSampler import TrainBalancedBatchSampler

import colored_traceback.always

onlyMandX = False


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

# epoch = 0
# def focal_loss(y_pred, y_true, gamma=0, eps=1e-7):
#     logit = F.softmax(y_pred, dim=-1)
#     logit = logit.clamp(eps, 1. - eps)

#     weights = torch.ones_like(y_true).float()
#     # weights[:,-1] = 0 # X class
#     weights[:,-1] = 0 if epoch < 15 else 1 # X class
#     # weights[:,-1] = np.min([1.,0.6 + 0.3 / 10 * (10-(epoch+1))]) # X class
#     loss = -1 * weights * y_true * torch.log(logit) # cross entropy
#     loss = loss * (1 - logit) ** gamma # focal loss

#     return loss.sum() / y_pred.shape[0]

# def ib_focal_loss(input_values, ib, gamma):
#     """Computes the ib focal loss"""
#     p = torch.exp(-input_values)
#     loss = (1 - p) ** gamma * input_values * ib
#     return loss.mean()


# switch_epoch = 100

# class IB_FocalLoss(nn.Module):
#     def __init__(self, weight=None, alpha=10000., gamma=0.):
#         super(IB_FocalLoss, self).__init__()
#         assert alpha > 0
#         self.alpha = alpha
#         self.epsilon = 0.001
#         self.weight = weight
#         self.gamma = gamma

#     def forward(self, input, target, features, num_classes=4):
#         grads = torch.sum(torch.abs(F.softmax(input, dim=1) - F.one_hot(target, num_classes)),1) # N * 1
#         ib = grads*(torch.sum(torch.abs(features),dim=1))
#         ib = self.alpha / (ib + self.epsilon)
#         if epoch < switch_epoch:
#             return F.cross_entropy(input, target, reduction='mean', weight=self.weight)
#         else:
#             return ib_focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), ib, self.gamma)


def label_smoothing(y_true, epsilon):
    """Return label smoothed vector"""
    x = y_true + epsilon
    x = x / (1 + epsilon * 4)
    return x


def bs_loss_function(y_pred, y_true):
    """Compute BSS loss"""
    tmp = y_pred - y_true
    tmp = torch.mul(tmp, tmp)
    tmp = torch.sum(tmp, dim=1)
    tmp = torch.mean(tmp)
    return tmp


def train_epoch(model, train_dl, epoch, lr, args):
    """Return train loss and score for train set"""
    model.train()
    predictions = []
    observations = []
    train_loss = 0
    n = 0
    for _, (x, y, feat, idx) in enumerate(tqdm(train_dl)):
        if onlyMandX:
            mask = (y[:, 2] == 1) + (y[:, 3] == 1)
            x = x[mask]
            feat = feat[mask]
            y = y[mask]
            if x.shape[0] == 0:
                continue

        if not args.without_schedule:
            adjust_learning_rate(optimizer, epoch, params["epochs"], lr, args)
        optimizer.zero_grad()
        if isinstance(model, FlareTransformerWithMultiPE):
            output, feat = model(x.cuda().to(torch.float), feat.cuda().to(torch.float), idx.cuda().to(torch.float))
        else:
            output, feat = model(x.cuda().to(torch.float), feat.cuda().to(torch.float))

        # ib_loss = ib(output, torch.max(y, 1)[1].cuda().to(torch.long),feat)
        bce_loss = criterion(output, torch.max(y, 1)[1].cuda().to(torch.long))

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
                       params["dataset"]["climatology"])
    score = calc_test_score(score, "train")

    return score, train_loss / n


def eval_epoch(model, validation_dl):
    """Return val loss and score for val set"""
    model.eval()
    predictions = []
    observations = []
    valid_loss = 0
    n = 0
    with torch.no_grad():
        for _, (x, y, feat, idx) in enumerate(tqdm(validation_dl)):
            if onlyMandX:
                mask = (y[:, 2] == 1) + (y[:, 3] == 1)
                x = x[mask]
                feat = feat[mask]
                y = y[mask]
                if x.shape[0] == 0:
                    continue

            if isinstance(model, FlareTransformerWithMultiPE):
                idx += len(train_dataset)
                output, feat = model(x.cuda().to(torch.float), feat.cuda().to(torch.float), idx.cuda().to(torch.float))
            else:
                output, feat = model(x.cuda().to(torch.float), feat.cuda().to(torch.float))

            # bce_loss = criterion(output, y.cuda().to(torch.long))
            bce_loss = criterion(output, torch.max(y, 1)[1].cuda().to(torch.long))
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
    return score, valid_loss / n


def calc_model_update(model, dl, model_update_dict):  # モデルの更新量を計算する
    model.eval()
    with torch.no_grad():
        for _, (x, y, feat, idx) in enumerate(tqdm(dl)):
            x = x.cuda().to(torch.float)
            feat = feat.cuda().to(torch.float)
            mm_extractor = model.forward_mm_feature_extractor
            sfm_extractor = model.forward_sfm_feature_extractor
            mm_output = mm_extractor(x, feat).cpu().numpy()  # (B, W, d_model)
            sfm_output = sfm_extractor(x, feat).cpu().numpy()
            # mm_mean = mm_output.cpu().numpy().mean(axis=0)
            # sfm_mean = sfm_output.cpu().numpy().mean(axis=0)

            # print(mm_output.shape)
            # print(sfm_output.shape)

            if "mm" not in model_update_dict:
                model_update_dict["mm"] = [mm_output]
            else:
                z = model_update_dict["mm"][0]
                z = z - mm_output
                t = [np.linalg.norm(z[i, 0, :]) for i in range(z.shape[0])]  # ウィンドウのheadだけ
                model_update_dict["mm"].append(np.mean(t))

            if "sfm" not in model_update_dict:
                model_update_dict["sfm"] = [sfm_output]
            else:
                z = model_update_dict["sfm"][0]
                z = z - sfm_output
                t = [np.linalg.norm(z[i, 0, :]) for i in range(z.shape[0])]  # ウィンドウのheadだけ
                model_update_dict["sfm"].append(np.mean(t))

            break
        # for pred, o in zip(output.cpu().numpy().tolist(),
        #                     y.numpy().tolist()):
        #     predictions.append(pred)
        #     observations.append(np.argmax(o))

    print("+-- model update --+")
    for k, v in model_update_dict.items():
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
        for _, (x, y, feat, idx) in enumerate(tqdm(test_dl)):
            if onlyMandX:
                mask = (y[:, 2] == 1) + (y[:, 3] == 1)
                x = x[mask]
                feat = feat[mask]
                y = y[mask]
                if x.shape[0] == 0:
                    continue

            if isinstance(model, FlareTransformerWithMultiPE):
                idx += len(train_dataset)
                output, feat = model(x.cuda().to(torch.float), feat.cuda().to(torch.float), idx.cuda().to(torch.float))
            else:
                output, feat = model(x.cuda().to(torch.float), feat.cuda().to(torch.float))

            # bce_loss = criterion(output, y.cuda().to(torch.long))
            bce_loss = criterion(output, torch.max(y, 1)[1].cuda().to(torch.long))
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
    return score, test_loss / n


def calc_test_score(score, label):
    """Return dict with key of label"""
    test_score = {}
    for k, v in score.items():
        test_score[label + "_" + k] = v
    return test_score


def adjust_learning_rate(optimizer, current_epoch, epochs, lr, args):  # optimizerの内部パラメタを直接変えちゃうので注意
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
    parser.add_argument('--warmup_epochs', default=5, type=int)
    parser.add_argument('--without_schedule', action='store_false')
    parser.add_argument('--lr_stage2', default=0.000008, type=float)
    parser.add_argument('--epoch_stage2', default=25, type=float)
    parser.add_argument('--imbalance', action='store_true')
    parser.add_argument('--debug_value', default=1.0, type=float)

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
    train_dataset = TrainDataloader256("train", params["dataset"], use_aia131=True, use_aia1600=True, aia_mix="mix")
    # aug_train_dataset = TrainDataloader("train", params["dataset"], augmentation=True)
    if params["dataset"]["mean"] != 0:
        mean = params["dataset"]["mean"]
        std = params["dataset"]["std"]
    else:
        mean, std = train_dataset.calc_mean()
        print(mean, std)

    mean, std = train_dataset.calc_mean()
    mean *= args.debug_value
    std *= args.debug_value
    print(mean, std)
    train_dataset.set_mean(mean, std)

    validation_dataset = TrainDataloader256("valid", params["dataset"], use_aia131=True, use_aia1600=True, aia_mix="mix")
    validation_dataset.set_mean(mean, std)
    test_dataset = TrainDataloader256("test", params["dataset"], use_aia131=True, use_aia1600=True, aia_mix="mix")
    test_dataset.set_mean(mean, std)

#############
    # classes = torch.argmax(torch.Tensor(test_dataset.label),dim=1)
    # prob = torch.zeros(4).cuda()
    # for i in range(test_dataset.label.shape[0]):
    #     prob[classes[i]] += 1

    # prob = 1 / prob
    # prob /= prob.sum()
    # print(prob)

#############

    print("Batch Sampling")

    if args.imbalance:
        train_dl = DataLoader(train_dataset, batch_size=params["bs"], shuffle=True, num_workers=2)
    else:
        train_dl = DataLoader(train_dataset, batch_sampler=TrainBalancedBatchSampler(
            train_dataset, params["output_channel"], params["bs"] // params["output_channel"]), num_workers=2)

    validation_dl = DataLoader(validation_dataset,
                               batch_size=params["bs"], shuffle=False, num_workers=2)
    test_dl = DataLoader(test_dataset, batch_size=params["bs"], shuffle=False, num_workers=2)

    # Initialize Loss Function
    criterion = nn.CrossEntropyLoss().cuda()
    # ib = IB_FocalLoss()
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

    # model = FlareTransformerAIA(input_channel=params["input_channel"],
    #                          output_channel=params["output_channel"],
    #                          sfm_params=params["SFM"],
    #                          mm_params=params["MM"],
    #                          window=params["dataset"]["window"]).to("cuda")

    # model = FlareTransformerWithPE(input_channel=params["input_channel"],
    #                             output_channel=params["output_channel"],
    #                             sfm_params=params["SFM"],
    #                             mm_params=params["MM"],
    #                             window=params["dataset"]["window"]).to("cuda")

    # model = FlareTransformerWithMultiPE(input_channel=params["input_channel"],
    #                         output_channel=params["output_channel"],
    #                         sfm_params=params["SFM"],
    #                         mm_params=params["MM"],
    #                         window=params["dataset"]["window"]).to("cuda")

    model = FlareTransformerWithConvNextAIA(input_channel=params["input_channel"],
                                            output_channel=params["output_channel"],
                                            input_conv_channel=1,
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

    summary(model)
    # summary(model,[(params["bs"], *train_dataset[0][0].shape),(params["bs"], *train_dataset[0][2].shape)])
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])

    # Start Training
    best_score = {}
    best_score["valid_" + params["main_metric"]] = -10
    best_epoch = 0
    model_update_dict = {}
    for e, epoch in enumerate(range(params["epochs"])):
        print("====== Epoch ", e, " ======")
        train_score, train_loss = train_epoch(model, train_dl, epoch, params["lr"], args)
        valid_score, valid_loss = eval_epoch(model, validation_dl)
        # test_score, test_loss = test_epoch(model, test_dl) # for train/val/test model
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

    # ここからCRT
    if args.imbalance:
        print("Start CRT")
        train_dl = DataLoader(train_dataset, batch_sampler=TrainBalancedBatchSampler(
            train_dataset, params["output_channel"], params["bs"] // params["output_channel"]))

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

            if wandb_flag is True:
                wandb.log(log)

            print("Epoch {}: Train loss:{:.4f}  Valid loss:{:.4f}".format(
                e, train_loss, valid_loss), test_score)
