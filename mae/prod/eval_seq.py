# $ python eval_mae.py --target=seq --baseline=attn --input_size=256 --epoch=100 --dim=128 --batch_size=32 --enc_depth=4 --dec_depth=4 --baseline=attn --target=seq --mask_ratio=0.75 --output_dir=output_dir

from mae.prod.eval import *
from train_mae import FashionMnistDataLoader
import argparse
import json
from mae.prod.datasets import *
from tqdm import tqdm
from typing import Iterable
import sys
import math
from mae.prod.datasets import TrainDataloader
import mae.prod.models_mae
import timm.optim.optim_factory as optim_factory
import argparse
import datetime
import numpy as np
import os
import time
from pathlib import Path 
from functools import partial
import torch.nn as nn

import torch
import torch.backends.cudnn as cudnn
import wandb
from torchinfo import summary

P = 0.75
def prepare_model(chkpt_dir,args):
    # build model
    model =  mae.prod.models_mae.SeqentialMaskedAutoencoderViT(embed_dim=args.dim,
                                                                baseline=args.baseline, # attn, lambda, linear
                                                                img_size=args.input_size,
                                                                depth=args.enc_depth,
                                                                decoder_depth=args.dec_depth,
                                                                norm_pix_loss=False,
                                                                patch_size=16,
                                                                num_heads=8, 
                                                                decoder_embed_dim=512,
                                                                decoder_num_heads=8,
                                                                mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),)
    # load model
    checkpoint = torch.load(chkpt_dir, map_location=torch.device('cuda'))
    msg = model.load_state_dict(checkpoint['model'], strict=True)
    model.cuda()
    return model


def run_one_image(img, img2, model,mean,std):
    x = torch.cat([torch.tensor(img).cuda().unsqueeze(0),torch.tensor(img2).cuda().unsqueeze(0)],dim=0)
    
    # make it a batch-like
    print(x.shape)
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nkhwc->nkchw', x)

    # run MAE
    loss, y, mask = model(x, mask_ratio=P)
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach()
    print("loss", loss)

    # visualize the mask
    mask = mask.detach()
    # (N, H*W, p*p*3)
    mask = mask.unsqueeze(-1).repeat(1, 1,
                                     model.patch_embed.patch_size[0]**2)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach()

    x = torch.einsum('nkchw->nkhwc', x)
    for i in range(2):
        x[:,i,:,:,:] = mean + std * x[:,i,:,:,:]
    
    y = mean + std * y
    # masked image
    im_masked = x[:,1,:,:,:] * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x[:,1,:,:,:] * (1 - mask) + y * mask

    # make the plt figure larger
    plt.rcParams['figure.figsize'] = [24, 24]

    img1, img2 = x[:,0,:,:,:][0].cpu(), x[:,1,:,:,:][0].cpu()

    plt.subplot(2, 4, 1)
    show_image(img1, "$x_{t-1}$")

    plt.subplot(2, 4, 2)
    show_image(img2, "$x_t$")

    plt.subplot(2, 4, 3)
    show_image(np.abs(img1-img2), "$ | x_t - x_{t-1}| $")
    
    plt.subplot(2, 4, 5)
    show_image(img2, "original")

    plt.subplot(2, 4, 6)
    show_image(im_masked[0].cpu(), "masked")

    plt.subplot(2, 4, 7)
    show_image(y[0].cpu(), "reconstruction")

    plt.subplot(2, 4, 8)
    show_image(im_paste[0].cpu(), "reconstruction + visible")

    plt.show()

def run(args):
    # dl = TrainDataloader()
    # img, _ = dl[0]
    # img = img.transpose((1, 2, 0))

    params = json.loads(open("params/params_2014.json").read())
    dl = TrainDataloader256("train", params["dataset"],has_window=False)
    mean,std = dl.calc_mean()
    dl.set_mean(mean,std)
    print(mean,std)

    dl2 = TrainDataloader256("test", params["dataset"],has_window=False)
    dl2.set_mean(mean,std)

    img, _ = dl[0]
    img = img.transpose(0,1).transpose(1,2)

    img_test1, _ = dl2[0]
    img_test2, _ = dl2[24]
    img_test1 = img_test1.transpose(0,1).transpose(1,2)
    img_test2 = img_test2.transpose(0,1).transpose(1,2)

    plt.rcParams['figure.figsize'] = [5, 5]
    show_image(torch.tensor(img))

    chkpt_dir = f'/home/initial/Dropbox/flare_transformer/output_dir/attn/checkpoint-100.pth'

    model_mae = prepare_model(chkpt_dir,args)
    print('Model loaded.')

    # make random mask reproducible (comment out to make it change)
    torch.manual_seed(2)
    print('MAE with pixel reconstruction:')
    # run_one_image(img, model_mae, mean,std)
    run_one_image(img_test1,img_test2,  model_mae, mean,std)