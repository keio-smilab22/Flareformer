# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
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
 
    optimizer.zero_grad()

    for data_iter_step, (samples, _,_) in enumerate(tqdm(data_loader)):
        if data_iter_step % accum_iter == 0:
            adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.cuda()
        # samples = samples.cpu()
        with torch.cuda.amp.autocast():
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()
        # import time
        # time.sleep(100000)

        lr = optimizer.param_groups[0]["lr"]
        if (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int(
                (data_iter_step / len(data_loader) + epoch) * 1000)
            # print("{:.20f}".format(lr))
            # if args.wandb: wandb.log({"epoch_1000x": epoch_1000x, "loss": loss_value_reduce})

        if args.batch_size_search:
            break

    # gather the stats from all processes
    return None, loss_value


def main(args, dataset_train):
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    SEED = 42
    np.random.seed(SEED) 
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    cudnn.benchmark = True

    # dataset_train = TrainDataloader()

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=1, rank=0, shuffle=True)
    print("Sampler_train = %s" % str(sampler_train))

    # os.makedirs(args.log_dir, exist_ok=True)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # define the model
    model = mae.prod.models_mae.__dict__[args.model](embed_dim=args.dim,
                                                     baseline=args.baseline, # attn, lambda, linear
                                                     img_size=args.input_size,
                                                     depth=args.enc_depth,
                                                     decoder_depth=args.dec_depth,
                                                     norm_pix_loss=args.norm_pix_loss)
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

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    # print(optimizer)
    loss_scaler = NativeScaler()

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    if args.wandb:
        wandb.init(
            project="flare_transformer_MAE_exp",
            name=f"flare_{args.baseline}_b{args.batch_size}_dim{args.dim}_depth{args.enc_depth}-{args.dec_depth}")

    for epoch in range(args.epochs):
        print("====== Epoch ", (epoch+1), " ======")
        epoch_start = time.time()
        _, last_loss = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args=args
        )
        elapsed_epoch_time = time.time() - epoch_start
        print("loss: {:.3f}".format(last_loss))
        print("time:","{:.3f}s".format(elapsed_epoch_time))

        log = {'epoch': epoch, 'train_loss': last_loss}
        if args.wandb:
            wandb.log(log)

        if args.output_dir and ((epoch+1) == args.epoch or (epoch+1) == 1):
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


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
