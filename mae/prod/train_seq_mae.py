# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
from cgi import test
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


import timm
# assert timm.__version__ == "0.3.2"  # version check

import colored_traceback.always


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1. + math.cos(
            math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == math.inf:
        total_norm = max(p.grad.detach().abs().max().to(device)
                         for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(
            p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


class NativeScaler:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(
            self,
            loss,
            optimizer,
            clip_grad=None,
            parameters=None,
            create_graph=False,
            update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                # unscale the gradients of optimizer's assigned params in-place
                self._scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    args=None):
    model.train(True)
    accum_iter = args.accum_iter
 
    # optimizer.zero_grad()

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    for data_iter_step, (samples, _,_) in enumerate(tqdm(data_loader)):
        adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        optimizer.zero_grad()
        samples = samples.cuda()
        mk = torch.BoolTensor([i % args.interval == 0 for i in range(samples.shape[1])])
        
        assert mk.sum() == args.k
        with torch.cuda.amp.autocast(enabled=args.use_amp):
            loss, _, _ = model(samples[:,mk,:,:,:], mask_ratio=args.mask_ratio)


        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if args.use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if args.batch_size_search:
            break

    # gather the stats from all processes
    return None, loss_value


def eval_test(model: torch.nn.Module, data_loader: Iterable, args):
    model.eval()
    test_loss, n = 0, 0
    with torch.no_grad():
        for _, (samples, _, _) in enumerate(tqdm(data_loader)):
            samples = samples.cuda()
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio)
            loss_value = loss.item()
            test_loss += loss_value
            n += samples.shape[0]

    test_loss /= n
    return test_loss



def main(args, dataset_train, dataset_test):
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    SEED = 42
    np.random.seed(SEED) 
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    cudnn.benchmark = True

    # dataset_train = TrainDataloader()

    sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=1, rank=0, shuffle=True)
    sampler_test = torch.utils.data.DistributedSampler(dataset_test, num_replicas=1, rank=0, shuffle=True)
    # print("Sampler_train = %s" % str(sampler_train))

    # os.makedirs(args.log_dir, exist_ok=True)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=10,
        pin_memory=True,
        drop_last=True,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=10,
        pin_memory=True,
        drop_last=True,
    )

    # define the model
    model = mae.prod.models_mae.SeqentialMaskedAutoencoderConcatVersion(embed_dim=args.dim,
                                                            baseline=args.baseline, # attn, lambda, linear
                                                            img_size=args.input_size,
                                                            depth=args.enc_depth,
                                                            decoder_depth=args.dec_depth,
                                                            norm_pix_loss=False,
                                                            patch_size=args.patch_size,
                                                            num_heads=8,
                                                            decoder_embed_dim=512,
                                                            decoder_num_heads=8,
                                                            in_chans=args.k,
                                                            mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                                            mask_ratio=args.mask_ratio,
                                                            mask_token_type=args.mask_token_type)


    # model = mae.prod.models_mae.SeqentialMaskedAutoencoderViT(embed_dim=args.dim,
    #                                                         baseline=args.baseline, # attn, lambda, linear
    #                                                         img_size=args.input_size,
    #                                                         depth=args.enc_depth,
    #                                                         decoder_depth=args.dec_depth,
    #                                                         norm_pix_loss=False,
    #                                                         patch_size=args.patch_size,
    #                                                         num_heads=8, 
    #                                                         decoder_embed_dim=512,
    #                                                         decoder_num_heads=8,
    #                                                         mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),)
        
    model.to(device)

    model_without_ddp = model
    # print("Model = %s" % str(model_without_ddp))
    summary(model)

    eff_batch_size = args.batch_size * args.accum_iter

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(
        model_without_ddp, args.weight_decay)

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95),amsgrad=True)
    # print(optimizer)
    loss_scaler = NativeScaler()

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    if args.wandb:
        wandb.init(
            project="flare_transformer_MAE_exp",
            name=f"informer_{args.baseline}_b{args.batch_size}_dim{args.dim}_depth{args.enc_depth}-{args.dec_depth}_k={args.k}")

    for epoch in range(args.epochs):
        print("====== Epoch ", (epoch+1), " ======")
        epoch_start = time.time()
        _, last_loss = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args=args
        )
        elapsed_epoch_time = time.time() - epoch_start
        print("loss: {:.8f}".format(last_loss))
        print("time:","{:.3f}s".format(elapsed_epoch_time))

        # test_loss = eval_test(model,data_loader_test)
        # log = {'epoch': epoch, 'train_loss': last_loss, "test_loss" : test_loss}

        log = {'epoch': epoch, 'train_loss': last_loss}
        if args.wandb:
            wandb.log(log)

        saved_flag = False
        for i in range(5):
            saved_flag = saved_flag or (epoch+1) == args.epochs // (i+1) or epoch == 1
        
        if args.output_dir and saved_flag:
            output_dir = Path(args.output_dir)
            epoch_name = str(epoch+1)
            if loss_scaler is not None:
                checkpoint_paths = [output_dir / args.baseline / 
                                    ('checkpoint-%s.pth' % epoch_name)]
                for checkpoint_path in checkpoint_paths:
                    to_save = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'scaler': loss_scaler.state_dict(),
                        'args': args,
                    }

                    torch.save(to_save, checkpoint_path)
            else:
                client_state = {'epoch': epoch}
                model.save_checkpoint(
                    save_dir=args.output_dir,
                    tag="checkpoint-%s" %
                    epoch_name,
                    client_state=client_state)

            

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))