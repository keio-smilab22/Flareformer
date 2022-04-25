from functools import partial
 
import torch
import torch.nn as nn
import numpy as np
from typing import Type, Any, Callable, Union, List, Optional

from timm.models.vision_transformer import PatchEmbed, Mlp, DropPath, Attention


from mae.prod.util.pos_embed import get_2d_sincos_pos_embed


import torch
from torch import nn
import torch.nn.functional as F

from timm.models.layers.helpers import to_2tuple, make_divisible
from timm.models.layers.weight_init import trunc_normal_

from mae.prod.models_mae import *


class PyramidMaskedAutoencoderViT(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_chans=1,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, baseline="attn",
                 grid_size=128):
        super(PyramidMaskedAutoencoderViT, self).__init__()

        self.mae = MaskedAutoencoderViT(
            grid_size, patch_size, in_chans, embed_dim, depth//4, num_heads,
            decoder_embed_dim, decoder_depth//4, decoder_num_heads,
            mlp_ratio, norm_layer, norm_pix_loss, baseline)

        # load pretrained weights
        # ckpt = torch.load("/home/katsuyuki/temp/flare_transformer/output_dir/attn/checkpoint-50-pretrain-128-0.1.pth")
        # self.mae.load_state_dict(ckpt["model"])

        # freeze all layers
        # for p in self.mae.parameters():
        #     p.requires_grad = False

        self.mae2 = MaskedAutoencoderViT(
            img_size, patch_size, in_chans, embed_dim, depth, num_heads,
            decoder_embed_dim, decoder_depth, decoder_num_heads,
            mlp_ratio, norm_layer, norm_pix_loss, baseline)

        self.gird_size = grid_size
        
    def grid_dividing_image(self, x:torch.Tensor, rows:int, cols:int)-> List[torch.Tensor]:
        """
        x: [N, C, H, W]
        rows: int
        cols: int
        x_list: [N, C, H//rows, W//cols] * rows * cols
        """

        # x_list = []
        # for i in range(rows):
        #     for j in range(cols):
        #         x_list.append(x[:, :, i*x.shape[2]//rows:(i+1)*x.shape[2]//rows, j*x.shape[3]//cols:(j+1)*x.shape[3]//cols])
        #         # print(f"x_list[{i*cols+j}].shape: {x_list[i*cols+j].shape}")
        # return x_list

        x_list = []
        for row in torch.tensor_split(x, rows, dim=2):
            for col in torch.tensor_split(row, cols, dim=3):
                x_list.append(col)
                # print(f"x_list[{len(x_list)-1}].shape: {x_list[-1].shape}")
        return x_list

    def grid_merging_image(self, x_list:List[torch.Tensor], rows:int, cols:int)->torch.Tensor:
        """
        x_list: [N, C, H//rows, W//cols] * rows * cols
        x : [N, C, H, W]
        """
        # print(f"x_list[0].shape: {x_list[0].shape}")
        x = torch.zeros((x_list[0].shape[0], x_list[0].shape[1], rows*x_list[0].shape[2], cols*x_list[0].shape[3]), dtype=x_list[0].dtype, device=x_list[0].device)
        for i in range(rows):
            for j in range(cols):
                x[:, :, i*x_list[0].shape[2]:(i+1)*x_list[0].shape[2], j*x_list[0].shape[3]:(j+1)*x_list[0].shape[3]] = x_list[i*cols+j]
        return x

    def grid_merging_mask(self, x_list:List[torch.Tensor], rows:int, cols:int)->torch.Tensor:
        """
        x_list: [N, L] * rows * cols
        """
        # print(f"x_list[0].shape: {x_list[0].shape}")
        x = torch.zeros((x_list[0].shape[0], x_list[0].shape[1], rows, cols), dtype=x_list[0].dtype, device=x_list[0].device)
        for i in range(rows):
            for j in range(cols):
                x[:, :, i, j] = x_list[i*cols+j]
        return x

    def forward(self, imgs, mask_ratio=0.75, mask_ratio2=0.75):
        # print(f"imgs.shape: {imgs.shape}")
        h = imgs.shape[2]//self.gird_size
        w = imgs.shape[3]//self.gird_size

        imgs_list = self.grid_dividing_image(imgs, h, w)
        loss_list = []
        pred_list = []
        mask_list = []
        for i, img in enumerate(imgs_list):
            l, p, m = self.mae(img, mask_ratio)
            p = self.mae.unpatchify(p)
            loss_list.append(l)
            pred_list.append(p)
            mask_list.append(m)
        pred_merged = self.grid_merging_image(pred_list, h, w)
        loss = torch.mean(torch.stack(loss_list))

        # loss_merged = self.grid_merging_image(loss_list, self.gird_size, self.gird_size)
        mask_merged = self.grid_merging_mask(mask_list, h, w)

        # loss, pred, mask = self.mae2(pred_merged, mask_ratio2)
        latent, mask, ids_restore = self.mae2.forward_encoder(pred_merged, mask_ratio)
        pred = self.mae2.forward_decoder(latent, ids_restore)
        loss = self.mae2.forward_loss(imgs, pred, mask)

        
        return loss, pred_merged, mask_merged

def vit_for_FT64d4b(embed_dim=64, **kwargs):
    model = PyramidMaskedAutoencoderViT(
        patch_size=8, embed_dim=embed_dim, depth=4, num_heads=8, # embed_dim % num_heads == 0 にしないとだめなので注意
        decoder_embed_dim=128, decoder_depth=4, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

vit_for_FT = vit_for_FT64d4b