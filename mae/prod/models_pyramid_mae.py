from functools import partial
from sklearn.feature_extraction import img_to_graph
 
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Type, Any, Callable, Union, List, Optional

from timm.models.vision_transformer import PatchEmbed, Mlp, DropPath, Attention


from mae.prod.util.pos_embed import get_2d_sincos_pos_embed


import torch
from torch import chunk, nn
import torch.nn.functional as F

from timm.models.layers.helpers import to_2tuple, make_divisible
from timm.models.layers.weight_init import trunc_normal_

from mae.prod.models_mae import *


class PyramidMaskedAutoencoderViT(nn.Module):
    def __init__(self, img_size=256, patch_size=2, in_chans=1,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, baseline="attn",
                 grid_size=128):
        super(PyramidMaskedAutoencoderViT, self).__init__()

        self.mae = MaskedAutoencoderViT(
            grid_size, patch_size, in_chans, embed_dim, 1, num_heads,
            decoder_embed_dim, 1, decoder_num_heads,
            mlp_ratio, norm_layer, norm_pix_loss, baseline)

        # load pretrained weights
        # ckpt = torch.load("/home/katsuyuki/temp/flare_transformer/output_dir/attn/checkpoint-50-pretrain-128-0.1.pth")
        # self.mae.load_state_dict(ckpt["model"])

        # freeze all layers
        # for p in self.mae.parameters():
        #     p.requires_grad = False

        self.mae2 = MaskedAutoencoderViT(
            img_size, grid_size, in_chans, embed_dim, depth, num_heads,
            decoder_embed_dim, decoder_depth, decoder_num_heads,
            mlp_ratio, norm_layer, norm_pix_loss, baseline)

        self.grid_size = grid_size
        self.rows = img_size // grid_size
        self.cols = img_size // grid_size

        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, patch_size**2))

        
    def unpatchify(self, x:torch.Tensor): # チャネル数1
        """
        x: (G, N, L, patch_size**2)
        imgs: (G, N, H//rows, W//cols)
        """
        imgs = []
        for i in range(x.shape[0]):
            img = self.mae.unpatchify(x[i])
            imgs.append(img)
        return torch.stack(imgs, dim=0)
        
        p = self.mae.patch_embed.patch_size[0]
        rows = self.rows
        cols = self.cols

        # print(f"x.shape: {x.shape}")
        h = w = int(x.shape[2]**.5) 
        assert h * w == x.shape[2]
        
        imgs = torch.zeros((x.shape[1], 1, self.grid_size*rows, self.grid_size*cols), dtype=x.dtype, device=x.device)

        for i in range(rows):
            for j in range(cols):
                x_patch = x[i*cols+j]
                img = self.mae.unpatchify(x_patch)
                imgs[:, :, i*self.grid_size:(i+1)*self.grid_size, j*self.grid_size:(j+1)*self.grid_size] = img
        return imgs
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 1, h * p, h * p))
        return imgs
        
    def grid_dividing_image(self, x:torch.Tensor, rows:int, cols:int)-> List[torch.Tensor]:
        """
        x: [N, C, H, W]
        rows: int
        cols: int
        x_list: [rows*cols, N, C, H//rows, W//cols]
        """

        # x_list = []
        # for i in range(rows):
        #     for j in range(cols):
        #         x_list.append(x[:, :, i*x.shape[2]//rows:(i+1)*x.shape[2]//rows, j*x.shape[3]//cols:(j+1)*x.shape[3]//cols])
        #         # print(f"x_list[{i*cols+j}].shape: {x_list[i*cols+j].shape}")
        # return x_list
        
        # divide image into grids
        # chunks = torch.chunk(x, rows, dim=2)
        # chunks = [torch.chunk(chunk, cols, dim=3) for chunk in chunks]

        x_list = []
        std_list = []
        for row in torch.tensor_split(x, rows, dim=2):
            for col in torch.tensor_split(row, cols, dim=3):
                x_list.append(col)
                std = col.std(dim=(1,2,3), keepdim=False, unbiased=False)
                std_list.append(std)
                # print(f"x_list[{len(x_list)-1}].shape: {x_list[-1].shape}")
        xs = torch.stack(x_list, dim=0)
        stds = torch.stack(std_list, dim=0)
        # print(f"xs.shape: {xs.shape}")
        # print(f"stds.shape: {stds.shape}")
        return xs, stds

    def grid_merging_image(self, x_list:torch.Tensor, rows:int, cols:int, ids_restore:torch.Tensor)->torch.Tensor:
        """
        x_list: [G, N, L, D]
        x : [N, L*rows*cols, D]
        """
        x = torch.zeros((x_list.shape[1], x_list.shape[2]*rows*cols, x_list.shape[3]), dtype=x_list.dtype, device=x_list.device)
        
        # print(f"x.shape: {x.shape}")
        for i in range(rows):
            for j in range(cols):
                x[:, i*cols*x_list.shape[2]+j*x_list.shape[2]:i*cols*x_list.shape[2]+(j+1)*x_list.shape[2], :] += x_list[i*cols+j]
        return x

        x = torch.zeros((x_list[0].shape[0], x_list[0].shape[1]*rows*cols, x_list[0].shape[2]), dtype=x_list[0].dtype, device=x_list[0].device)
        for i in range(rows):
            for j in range(cols):
                x[:, i*cols*x_list[0].shape[1]+j*x_list[0].shape[1]:i*cols*x_list[0].shape[1]+(j+1)*x_list[0].shape[1], :] = x_list[i*cols+j]
        return x

    def grid_merging_mask(self, x_list:List[torch.Tensor], rows:int, cols:int)->torch.Tensor:
        """
        x_list: [N, L] * rows * cols
        """
        # print(f"x_list[0].shape: {x_list[0].shape}")
        x = torch.zeros((x_list.shape[0], x_list.shape[1], rows, cols), dtype=x_list[0].dtype, device=x_list[0].device)
        for i in range(rows):
            for j in range(cols):
                x[:, :, i, j] = x_list[i*cols+j]
        return x

    def grid_merging_loss(self, x_list:List[torch.Tensor], rows:int, cols:int)->torch.Tensor:
        """
        x_list: [] * rows * cols
        """
        
        x = torch.zeros((x_list[0].shape[0], rows, cols), dtype=x_list[0].dtype, device=x_list[0].device)
        for i in range(rows):
            for j in range(cols):
                x[:, i, j] = x_list[i*cols+j]
        return x

    def unsuffle_image(self, x_list:torch.Tensor, rows:int, cols:int, ids_restore:torch.Tensor)->torch.Tensor:
        """
        x_list: [G_mask, N, L, D]
        x : [G, N, L, D]
        """
        # imgs = torch.zeros((x.shape[1], 1, self.grid_size*rows, self.grid_size*cols), dtype=x.dtype, device=x.device)
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            ids_restore.shape[0] - x_list.shape[0], x_list.shape[1], x_list.shape[2], 1)
        # print(f"mask_tokens.shape: {mask_tokens.shape}")
        # print(f"x.shape: {x_list.shape}")
        x_list = torch.cat([x_list, mask_tokens], dim=0)  # no cls token
        # print(f"x.shape: {x_list.shape}")
        x_list = torch.gather(x_list, 
                          dim=0,
                          index=ids_restore.unsqueeze(-1).unsqueeze(-1).repeat((1, 1, x_list.shape[2], x_list.shape[3]))) # unshuffle
        return x_list

    def reshape_image(self, x_list:torch.Tensor, rows:int, cols:int)->torch.Tensor:
        """
        x_list: [G, N, C, H//rows, W//cols]
        x: [N, C, H, W]
        """
        x = torch.zeros((x_list.shape[1], x_list.shape[2], x_list.shape[3]*rows, x_list.shape[4]*cols), dtype=x_list.dtype, device=x_list.device)
        for i in range(rows):
            for j in range(cols):
                x[:, :, i*x_list.shape[3]:(i+1)*x_list.shape[3], j*x_list.shape[4]:(j+1)*x_list.shape[4]] = x_list[i*cols+j]
        return x


    def forward_encoder(self, imgs_list:List[torch.Tensor], mask_ratio:float=0.5)->Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        x: [N, C, H, W]
        mask_ratio: float
        """
        mask_list = []
        latent_list = []
        ids_restore_list = []
        for i in range(imgs_list.shape[0]):
            img = imgs_list[i]
            latent, m, ids_restore = self.mae.forward_encoder(img, mask_ratio=mask_ratio)
            mask_list.append(m)
            latent_list.append(latent)
            ids_restore_list.append(ids_restore)
        mask_list = torch.stack(mask_list, dim=0)
        latent_list = torch.stack(latent_list, dim=0)
        ids_restore_list = torch.stack(ids_restore_list, dim=0)
        
        return latent_list, mask_list, ids_restore_list


        for i, img in enumerate(imgs_list):
            img = img.to(self.mae.device)
            latent, m, ids_restore = self.mae.forward_encoder(img, mask_ratio=mask_ratio)
            mask_list.append(m)
            latent_list.append(latent)
            ids_restore_list.append(ids_restore)
        

        return latent_list, mask_list, ids_restore_list

    def forward_decoder(self, latent_list:List[torch.Tensor], mask_list:List[torch.Tensor], ids_restore_list:List[torch.Tensor])->torch.Tensor:
        """
        """
        rows = int((len(latent_list)**0.5))
        cols = int((len(latent_list)**0.5))
        # print(f"rows: {rows}, cols: {cols}")
        pred_list = []
        for i in range(latent_list.shape[0]):
            latent = latent_list[i]
            # mask = mask_list[i]
            ids_restore = ids_restore_list[i]
            pred = self.mae.forward_decoder(x=latent, ids_restore=ids_restore)
            pred_list.append(pred)
        pred_list = torch.stack(pred_list, dim=0)
        return pred_list

        for i, latent in enumerate(latent_list):
            latent = latent.to(self.mae.device)
            pred = self.mae.forward_decoder(latent, ids_restore_list[i])
            pred_list.append(pred)
        
        # pred = self.grid_merging_image(pred_list, rows=rows, cols=cols)
        return pred_list

    def forward_loss(self, imgs_list, pred_list, mask_list)->torch.Tensor:
        """
        """
        loss_list = []
        for i in range(imgs_list.shape[0]):
            img = imgs_list[i]
            pred = pred_list[i]
            mask = mask_list[i]
            loss = self.mae.forward_loss(img, pred, mask)
            loss_list.append(loss)
        loss_list = torch.stack(loss_list, dim=0)
        return loss_list

        for i, pred in enumerate(pred_list):
            pred = pred.to(self.mae.device)
            mask = mask_list[i]
            # print(f"pred.shape: {pred.shape}, imgs_list[i].shape: {imgs_list[i].shape}, mask.shape: {mask.shape}")
            loss = self.mae.forward_loss(imgs_list[i], pred, mask)
            loss_list.append(loss)
        return loss_list

    def std_masking(self, imgs:torch.Tensor, stds:torch.Tensor, keep_ratio:float):
        """
        Extract only the N% of the total images that have a large standard deviation.
        """

        G, N, C, H, W = imgs.shape
        # sort stds in descending order
        # print(f"std_list: {std_list.shape}")
        ids_shuffle = torch.argsort(stds, dim=0, descending=True)
        ids_restore = torch.argsort(ids_shuffle, dim=0)

        len_keep = int(G * (keep_ratio))
        ids_keep = ids_shuffle[:len_keep, :]
        
        imgs = torch.gather(imgs, dim=0, index=ids_keep.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, imgs.shape[2], imgs.shape[3], imgs.shape[4]))
        
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([G, N, C, H, W], device=imgs.device)
        mask[:len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=0, index=ids_keep.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, imgs.shape[2], imgs.shape[3], imgs.shape[4]))

        return imgs, mask, ids_restore

    def random_masking(self, imgs:torch.Tensor, mask_ratio:float):
        
        G, N, C, H, W = imgs.shape
        len_keep = int(G * (1 - mask_ratio))

        noise = torch.rand(G, N, device=imgs.device)

        ids_shuffle = torch.argsort(noise, dim=0, descending=True)
        ids_restore = torch.argsort(ids_shuffle, dim=0)

        ids_keep = ids_shuffle[:len_keep, :]

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([G, N, C, H, W], device=imgs.device)
        mask[:len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=0, index=ids_keep.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, imgs.shape[2], imgs.shape[3], imgs.shape[4]))
        imgs = torch.gather(imgs, dim=0, index=ids_keep.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, imgs.shape[2], imgs.shape[3], imgs.shape[4]))

        return imgs, mask, ids_restore




    def forward(self, imgs, mask_ratio=0.5, mask_ratio2=0.5, keep_ratio=0.75):
        
        rows = imgs.shape[2]//self.grid_size
        cols = imgs.shape[3]//self.grid_size

        imgs_list, std_list = self.grid_dividing_image(imgs, rows=rows, cols=cols)
        
        # imgs_list, mask_std, ids_restore_std = self.std_masking(imgs_list, std_list, keep_ratio=keep_ratio)
        
        latent_list, mask_list, ids_restore_list = self.forward_encoder(imgs_list, mask_ratio=mask_ratio)
        pred_list = self.forward_decoder(latent_list=latent_list, mask_list=mask_list, ids_restore_list=ids_restore_list)

        loss_list = self.forward_loss(imgs_list, pred_list=pred_list, mask_list=mask_list)

        loss = torch.mean(loss_list)
        
        # if use second stage
        img_merged = self.unpatchify(pred_list)
        img_merged = self.reshape_image(img_merged, rows=rows, cols=cols)
        img_merged = img_merged.to(imgs.device)

        latent, mask, ids_restore = self.mae2.forward_encoder(img_merged, mask_ratio=mask_ratio2)
        pred = self.mae2.forward_decoder(latent, ids_restore)

        loss2 = self.mae2.forward_loss(imgs, pred, mask)

        # loss_l = loss
        mask_merged = mask_list

        coef = 0.5
        loss = coef*loss + (1-coef)*loss2

        mask = mask
        
        return loss, pred, mask, ids_restore

def vit_for_FT64d4b(embed_dim=64, grid_size=128, **kwargs):
    model = PyramidMaskedAutoencoderViT(
        patch_size=4, embed_dim=embed_dim, depth=1, num_heads=8, # embed_dim % num_heads == 0 にしないとだめなので注意
        decoder_embed_dim=128, decoder_depth=1, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), grid_size=grid_size, **kwargs)
    return model

vit_for_FT = vit_for_FT64d4b