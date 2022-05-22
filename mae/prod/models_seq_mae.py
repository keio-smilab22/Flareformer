# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
from xml.etree.ElementPath import xpath_tokenizer
 
import torch
import torch.nn as nn
import numpy as np
from typing import Type, Any, Callable, Union, List, Optional

from timm.models.vision_transformer import PatchEmbed, Mlp, DropPath, Attention


from mae.prod.util.pos_embed import get_2d_sincos_pos_embed, get_2d_sincos_pos_embed_seq
from mae.prod.models_mae import LambdaResnet, Block

import torch
from torch import nn
import torch.nn.functional as F

from timm.models.layers.helpers import to_2tuple, make_divisible
from timm.models.layers.weight_init import trunc_normal_

class SequenceMaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    # 512の場合はここ変える

    def __init__(self, img_size=256, patch_size=16, in_chans=1,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, baseline="attn"):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed1 = PatchEmbed(
            img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed1.num_patches

        self.patch_embed2 = PatchEmbed(
            img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed2.num_patches

        self.embed_dim = embed_dim
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(
                1,
                num_patches * 2 + 1,
                embed_dim),
            requires_grad=False)  # fixed sin-cos embedding

        if baseline == "lambda_resnet":
            self.blocks = nn.ModuleList([
                    LambdaResnet(embed_dim), # todo: 128→embed_dimに修正
                    # Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,baseline=baseline)
                    ])
        else:
            self.blocks = nn.ModuleList([
                    Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,baseline=baseline)
                    for i in range(depth)])

        self.baseline = baseline
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(
                1,
                num_patches + 1,
                decoder_embed_dim),
            requires_grad=False)  # fixed sin-cos embedding

        decoder_baseline = baseline if baseline != "lambda_resnet" else "attn"
        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    baseline=decoder_baseline) for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            patch_size**2 * in_chans,
            bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def set_train_flag_encoeder(self):
        self.patch_embed.train()
        # self.cls_token.train()
        # self.pos_embed.train()
        self.blocks.train()
        self.norm.train()

        # decoder_modules = [self.decoder_embed,
        # self.decoder_blocks,
        # self.decoder_norm,
        # self.decoder_pred]

        # for module in decoder_modules:
        #     module.requires_grad = False
        #     del module
        # import gc
        # gc.collect()


        del self.decoder_embed
        del self.decoder_blocks
        del self.decoder_norm
        del self.decoder_pred
        import gc
        gc.collect()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed_seq(
            self.pos_embed.shape[-1], int((self.patch_embed1.num_patches** .5)), cls_token=True)
        
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], int((self.patch_embed1.num_patches** .5)), cls_token=True)
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed1.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as
        # cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def patchify(self, imgs):  # チャネル数1
        """
        imgs: (N, 1, H, W)
        x: (N, L, patch_size**2 *1)
        """
        p = self.patch_embed1.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 1))
        return x

    def unpatchify(self, x): # チャネル数1
        """
        x: (N, L, patch_size**2 *1)
        imgs: (N, 1, H, W)
        """
        p = self.patch_embed1.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 1, h * p, h * p))
        return imgs

    def patchify_dim3(self, imgs):  # 別にmaskとかは関係なく, 次元を変えてる
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed1.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify_dim3(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed1.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking_vit(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        # ascend: small is keep, large is remove
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def random_pyramid_masking_vit(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        # ascend: small is keep, large is remove
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        ids_not_keep = ids_shuffle[:, len_keep:]

        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        x_not_masked = torch.gather(x, dim=1, index=ids_not_keep.unsqueeze(-1).repeat(1, 1, D))

        noise_dim = torch.rand(D, device=x.device)  # noise in [0, 1]
        ids_shuffle_dim = torch.argsort(noise_dim, dim=0)
        ids_restore_dim = torch.argsort(ids_shuffle_dim, dim=0)

        ids_keep_dim = ids_shuffle_dim[:len_keep]
        ids_not_keep_dim = ids_shuffle_dim[len_keep:]

        x_not_masked[:,:,ids_not_keep_dim] = 0
        
        # x = torch.cat([x_masked, x_not_masked], dim=1)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x, mask, ids_restore

    def random_masking_resnet(self, x, mask_ratio):
        B,C,H,W = x.shape
        L = H*W
        x = x.view(B,C,L).transpose(-1,-2) # (B,L,C)
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(B, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        # ascend: small is keep, large is remove
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, C))

        B,l,C = x_masked.shape
        h = int(np.sqrt(l))
        w = l // h
        assert h == w
        x_masked = x_masked.transpose(-1,-2).view(B,C,h,w)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore


    def forward_encoder_vit(self, x, mask_ratio):  # x: (B,C,H,W)
        # print("base",x.shape)
        # embed patches
        # print(f"x.shape: {x.shape}")
        x_t = x[:, 0, :, :, :]
        x_t_1 = x[:, -1, :, :, :]
        x_t = self.patch_embed1(x_t) # (B,H*W/patch**2,embed_dim)
        x_t_1 = self.patch_embed2(x_t_1) # (B,H*W/patch**2,embed_dim)
        # print(x_t.shape)
        
        # add pos embed w/o cls token
        x_t = x_t + self.pos_embed[:, 1:x_t.shape[1]+1, :]
        x_t_1 = x_t_1 + self.pos_embed[:, x_t.shape[1]+1:, :]
        # print(x.shape)

        # masking: length -> length * mask_ratio
        x_t, mask, ids_restore = self.random_masking_vit(x_t, mask_ratio)
        # print(x.shape)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x_t.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x_t, x_t_1), dim=1)
        # print(f"x.shape: {x_t.shape}")
        # print(f"cls_token.shape: {cls_token.shape}")

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # print(x.shape)
        return x, mask, ids_restore

    def forward_encoder_vit_pyramid(self, x, mask_ratio):  # x: (B,C,H,W)
        # print("base",x.shape)
        # embed patches
        # print(x.shape)
        # 
        x = self.patch_embed(x) # (B,H*W/patch**2,embed_dim)
        # print(x.shape)
        
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        # print(x.shape)

        # masking: length -> length * mask_ratio
        # x, mask, ids_restore = self.random_masking_vit(x, mask_ratio)
        x, mask, ids_restore = self.random_pyramid_masking_vit(x, mask_ratio)
        # print(x.shape)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # print(x.shape)
        return x, mask, ids_restore

    def forward_encoder_lambda(self, x, mask_ratio):  # x: (B,C,H,W)
        # print("base",x.shape)
        # embed patches
        # print(x.shape)
        x = self.patch_embed(x) # (B,H*W/patch**2,embed_dim)
        # print(x.shape)
        
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        # print(x.shape)

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking_vit(x, mask_ratio)
        # print(x.shape)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        # print("encoder",x.shape)
        for blk in self.blocks:
            # print("original",x.shape)
            x = blk(x)
        x = self.norm(x)

        # print(x.shape)
        return x, mask, ids_restore
        
    def forward_encoder_lambda_resnet(self, x, mask_ratio):  # x: (B,C,H,W)
        # masking: length -> length * mask_ratio
        print(x.shape)
        x, mask, ids_restore = self.random_masking_resnet(x, mask_ratio)

        print(x.shape)
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        print(x.shape)
        x = x.unsqueeze(1)
        x = self.norm(x) # todo: これいる？

        # print(x.shape)
        return x, mask, ids_restore

    def forward_encoder(self,x,mask_ratio):
        if self.baseline == "attn":
            return self.forward_encoder_vit(x,mask_ratio)
        elif self.baseline == "lambda" or self.baseline == "linear":
            return self.forward_encoder_lambda(x,mask_ratio)
        elif self.baseline == "lambda_resnet":
            return self.forward_encoder_lambda_resnet(x,mask_ratio)
        else:
            assert False

    def forward_encoder_pyramid(self,x,mask_ratio):
        if self.baseline == "attn":
            return self.forward_encoder_vit_pyramid(x,mask_ratio)
        elif self.baseline == "lambda" or self.baseline == "linear":
            return self.forward_encoder_lambda(x,mask_ratio)
        elif self.baseline == "lambda_resnet":
            return self.forward_encoder_lambda_resnet(x,mask_ratio)
        else:
            assert False

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        # print(f"x.shape:{x.shape}")
        x = self.decoder_embed(x[:, :256, :])
        # x_t_1 = (x[:, 256:, :])


        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_,
                          dim=1,
                          index=ids_restore.unsqueeze(-1).repeat(1,
                                                                 1,
                                                                 x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        # print("decoder",x.shape)
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_decoder_pyramid(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        # mask_tokens = self.mask_token.repeat(
            # x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        # x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x,
                          dim=1,
                          index=ids_restore.unsqueeze(-1).repeat(1,
                                                                 1,
                                                                 x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        # print("decoder",x.shape)
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        img_t = imgs[:, 0, :, :, :]
        target = self.patchify(img_t)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        # print(imgs.shape)
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        # latent, mask, ids_restore = self.forward_encoder_pyramid(imgs, mask_ratio)
        # pred = self.forward_decoder_pyramid(latent, ids_restore)  # [N, L, p*p*3]

        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

def vit_for_FT64d4b(embed_dim=64, **kwargs):
    model = SequenceMaskedAutoencoderViT(
        patch_size=8, embed_dim=embed_dim, depth=4, num_heads=8, # embed_dim % num_heads == 0 にしないとだめなので注意
        decoder_embed_dim=128, decoder_depth=4, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

vit_for_FT = vit_for_FT64d4b

