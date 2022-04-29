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
 
import torch
import torch.nn as nn
import numpy as np
from typing import Type, Any, Callable, Union, List, Optional
from mae.prod.cosformer import CosformerAttention

from timm.models.vision_transformer import PatchEmbed, Mlp, DropPath, Attention


from mae.prod.util.pos_embed import get_2d_sincos_pos_embed, get_2d_sincos_pos_embed_ex


import torch
from torch import nn
import torch.nn.functional as F

from timm.models.layers.helpers import to_2tuple, make_divisible
from timm.models.layers.weight_init import trunc_normal_


def rel_pos_indices(size):
    size = to_2tuple(size)
    pos = torch.stack(torch.meshgrid(torch.arange(size[0]), torch.arange(size[1]))).flatten(1)
    rel_pos = pos[:, None, :] - pos[:, :, None]
    rel_pos[0] += size[0] - 1
    rel_pos[1] += size[1] - 1
    return rel_pos  # 2, H * W, H * W

class LambdaLayer(nn.Module):
    def __init__(
            self, dim, dim_out=None, feat_size=None, stride=1, num_heads=4, dim_head=16, r=9,
            qk_ratio=1.0, qkv_bias=False, linear=False):
        super().__init__()
        dim_out = dim_out or dim
        assert dim_out % num_heads == 0, ' should be divided by num_heads'
        self.dim_qk = dim_head or make_divisible(dim_out * qk_ratio, divisor=8) // num_heads
        self.num_heads = num_heads
        self.dim_v = dim_out // num_heads
        self.linear = linear

        self.qkv = nn.Conv2d(
            dim,
            num_heads * self.dim_qk + self.dim_qk + self.dim_v,
            kernel_size=1, bias=qkv_bias)
        self.norm_q = nn.BatchNorm2d(num_heads * self.dim_qk)
        self.norm_v = nn.BatchNorm2d(self.dim_v)
        self.conv_lambda = None

        if r is not None:
            # local lambda convolution for pos
            if not self.linear:
                self.conv_lambda = nn.Conv3d(1, self.dim_qk, (r, r, 1), padding=(r // 2, r // 2, 0))
            
            self.pos_emb = None
            self.rel_pos_indices = None
        else:
            # relative pos embedding
            assert feat_size is not None
            feat_size = to_2tuple(feat_size)
            rel_size = [2 * s - 1 for s in feat_size]
            self.conv_lambda = None
            self.pos_emb = nn.Parameter(torch.zeros(rel_size[0], rel_size[1], self.dim_qk))
            self.register_buffer('rel_pos_indices', rel_pos_indices(feat_size), persistent=False)

        self.pool = nn.AvgPool2d(2, 2) if stride == 2 else nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        trunc_normal_(self.qkv.weight, std=self.qkv.weight.shape[1] ** -0.5)  # fan-in
        if self.conv_lambda is not None:
            trunc_normal_(self.conv_lambda.weight, std=self.dim_qk ** -0.5)
        if self.pos_emb is not None:
            trunc_normal_(self.pos_emb, std=.02)

    def forward(self, x):
        B, C, H, W = x.shape
        M = H * W
        qkv = self.qkv(x)
        q, k, v = torch.split(qkv, [
            self.num_heads * self.dim_qk, self.dim_qk, self.dim_v], dim=1)
        q = self.norm_q(q).reshape(B, self.num_heads, self.dim_qk, M).transpose(-1, -2)  # B, num_heads, M, K
        v = self.norm_v(v).reshape(B, self.dim_v, M).transpose(-1, -2)  # B, M, V
        k = F.softmax(k.reshape(B, self.dim_qk, M), dim=-1)  # B, K, M

        content_lam = k @ v  # B, K, V
        content_out = q @ content_lam.unsqueeze(1)  # B, num_heads, M, V

        if self.linear:
            out = content_out.transpose(-1, -2).reshape(B, C, H, W)  # B, C (num_heads * V), H, W
            out = self.pool(out)
            return out

        if self.pos_emb is None:
            position_lam = self.conv_lambda(v.reshape(B, 1, H, W, self.dim_v))  # B, H, W, V, K
            position_lam = position_lam.reshape(B, 1, self.dim_qk, H * W, self.dim_v).transpose(2, 3)  # B, 1, M, K, V
        else:
            # FIXME relative pos embedding path not fully verified
            pos_emb = self.pos_emb[self.rel_pos_indices[0], self.rel_pos_indices[1]].expand(B, -1, -1, -1)
            position_lam = (pos_emb.transpose(-1, -2) @ v.unsqueeze(1)).unsqueeze(1)  # B, 1, M, K, V
        position_out = (q.unsqueeze(-2) @ position_lam).squeeze(-2)  # B, num_heads, M, V

        out = (content_out + position_out).transpose(-1, -2).reshape(B, C, H, W)  # B, C (num_heads * V), H, W
        out = self.pool(out)
        return out


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class Block(nn.Module):

    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,baseline="attn"):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if baseline == "attn":
            self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        elif baseline == "lambda" or baseline == "linear":
            self.attn = LambdaLayer(dim, num_heads=num_heads, qk_ratio=1.0,linear=(baseline == "linear"))
        else:
            assert False

        self.baseline = baseline
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        if self.baseline == "attn":
            x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        
        elif self.baseline == "lambda" or self.baseline == "linear":
            B,L,D = x.shape
            z = self.norm1(x)
            z = z.transpose(-1,-2).unsqueeze(2)
            z = self.attn(z)
            z = z.squeeze(2).transpose(-1,-2)

            x = x + self.drop_path1(self.ls1(z))
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

        else:
            assert False
            
        return x



def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


class LambdaBottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        num_heads: int = 8,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.lamb = LambdaLayer(width, num_heads=num_heads, qk_ratio=1.0,linear=False)
        # self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # out = self.conv2(out)
        out = self.lamb(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class LambdaResnet(nn.Module): # train.pyのCNNModel参考
    def __init__(self, output_channel=4, size=2, pretrain=False):
        super().__init__()

        self.pretrain = pretrain
        self.conv1 = nn.Conv2d(1, 16, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        downsample = nn.Sequential(
            conv1x1(16, 8 * 4, 1),
            nn.BatchNorm2d(8 * 4),
        )
        self.layer1 = LambdaBottleneck(
            16, 8, 1, downsample=downsample, norm_layer=nn.BatchNorm2d
        )

        self.avgpool = nn.AdaptiveAvgPool2d((size, size))
        self.flatten = nn.Flatten()
        # self.fc = nn.Linear(32*size*size, output_channel)
        self.softmax = nn.Softmax(dim=1)

        self.fc = nn.Linear(32*size*size, 32)
        self.fc2 = nn.Linear(32, output_channel)
        self.bn3 = nn.BatchNorm2d(8 * 4)
        self.dropout = nn.Dropout()

    def forward(self, x):
        # print(x.shape)  # [bs, 1, 512, 512]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print(x.shape)  # [bs, 16, 128, 128]
        x = self.layer1(x)
        # print(x.shape)  # [bs, 32, 128, 128]
        x = self.avgpool(x)
        x = self.flatten(x)
        # print(x.shape)  # [bs, 32*2*2]

        if not self.pretrain:
            return x  # [bs, 128]

        x = self.fc(x)

        x = self.dropout(x)
        x = self.relu(x)  # [bs, 32]
        x = self.fc2(x)
        x = self.softmax(x)

        return x



class MaskedAutoencoderViT(nn.Module):
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
        self.patch_embed = PatchEmbed(
            img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.embed_dim = embed_dim
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(
                1,
                num_patches + 1,
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
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
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
        p = self.patch_embed.patch_size[0]
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
        p = self.patch_embed.patch_size[0]
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
        p = self.patch_embed.patch_size[0]
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
        p = self.patch_embed.patch_size[0]
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

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

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

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
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
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask





class SeqentialMaskedAutoencoderViT(nn.Module):
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
        self.patch_embed = PatchEmbed(
            img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.embed_dim = embed_dim
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(
                1,
                num_patches,
                embed_dim),
            requires_grad=False) 

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

        self.mask_token1 = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.mask_token2 = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(
                1,
                num_patches * 2,
                decoder_embed_dim),
            requires_grad=False) 

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
        N = int(self.patch_embed.num_patches**.5)
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], N, cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed_ex(self.decoder_pos_embed.shape[-1], N * 2, N, cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as
        # cutoff is too big (2.)
        # torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token1, std=.02)
        torch.nn.init.normal_(self.mask_token2, std=.02)

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
        p = self.patch_embed.patch_size[0]
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
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 1, h * p, h * p))
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

        return x_masked, mask, ids_restore, ids_shuffle

    def forward_encoder(self,x,mask_ratio=None):
        # embed patches
        x = self.patch_embed(x) # (B,H*W/patch**2,embed_dim)
        
        # add pos embed w/o cls token
        x = x + self.pos_embed

        # masking: length -> length * mask_ratio
        if mask_ratio is not None:
            x, mask, ids_restore, ids_shuffle = self.random_masking_vit(x, mask_ratio)
        else:
            mask, ids_restore, ids_shuffle = None, None, None

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return x, mask, ids_restore, ids_shuffle
     
    def forward_decoder(self, x1, x2, ids_restore):
        l1 = x1.shape[1]
        # print(x1.shape,x2.shape)
        x = torch.cat([x1,x2],dim=1) # L方向

        # embed tokens
        x = self.decoder_embed(x)
        x1, x2 = x[:,:l1,:], x[:,l1:,:]

        # append mask tokens to sequence
        mask_tokens1 = self.mask_token1.repeat(
            x1.shape[0], ids_restore.shape[1] + 1 - x1.shape[1], 1)
        x_ = torch.cat([x1, mask_tokens1], dim=1)
        x_ = torch.gather(x_,
                          dim=1,
                          index=ids_restore.unsqueeze(-1).repeat(1,
                                                                 1,
                                                                 x1.shape[2]))  # unshuffle
        x1 = x_

        mask_tokens2 = self.mask_token2.repeat(
            x2.shape[0], ids_restore.shape[1] + 1 - x2.shape[1], 1)
        x_ = torch.cat([x2, mask_tokens2], dim=1)
        x_ = torch.gather(x_,
                          dim=1,
                          index=ids_restore.unsqueeze(-1).repeat(1,
                                                                 1,
                                                                 x2.shape[2]))  # unshuffle
        x2 = x_

        # print(x1.shape,x2.shape)
        x = torch.cat([x1,x2],dim=1) # L方向

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        #remove x1
        # x = x[:,l1:,:]
        B, L, D = x.shape
        x = x[:,L//2:,:]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        x1, x2 = imgs[:,0,:,:,:], imgs[:,-1,:,:,:]
        y1, _, _, _ = self.forward_encoder(x1,mask_ratio=None)
        y2, mask, ids_restore, ids_shuffle = self.forward_encoder(x2,mask_ratio=mask_ratio)

        # mask y1
        ids_keep = ids_shuffle[:, :y2.shape[-2]]
        y1 = torch.gather(y1, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, y1.shape[-1]))
        
        pred = self.forward_decoder(y1, y2, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(x2, pred, mask)
        return loss, pred, mask

    


###### SeqentialMAE

class LinearAttention_(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.elu = torch.nn.ELU()

    def phi(self, x): 
        return self.elu(x) + 1

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        # q = q * self.scale
        Q = q.clone()
        # attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.phi(k.transpose(-2, -1)) @ v * self.scale
        attn = self.phi(q) @ attn
        x = self.attn_drop(attn).transpose(1,2).reshape(B,N,C)

        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

# from src.model import InformerEncoderLayer, AttentionLayer, ProbAttention
from math import sqrt

class InformerEncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(InformerEncoderLayer, self).__init__()

        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model,
                               out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(
            in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, q, kv):
        """
        How to use'
            x = encoder_layer(x)
            (original) x, attn = attn_layer(x, attn_mask=attn_mask)
        """
        # x [B, L, D]
        # x = x + self.dropout(self.attention(
        #     x, x, x,
        #     attn_mask = attn_mask
        # ))
        new_x, attn = self.attention(
            q, kv, kv,
            attn_mask=None
        )
        q = q + self.dropout(new_x)

        y = q = self.norm1(q)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(q+y)  # , attn





class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(
            L_Q).unsqueeze(1), index_sample, :]
        # K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), :, :].squeeze(2)
        # print(Q.unsqueeze(-2).shape, K_sample.transpose(-2, -1).shape)
        # sys.exit()
        Q_K_sample = torch.matmul(
            Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)
        # print(Q_K_sample.shape)
        # sys.exit()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top
 
    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H,
                                                L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            # requires that L_Q == L_V, i.e. for self-attention only
            assert(L_Q == L_V)
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) /
                     L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[
                None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * \
            np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * \
            np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(
            queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1./sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(
            context, values, scores_top, index, L_Q, attn_mask)

        return context.transpose(2, 1).contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads,
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(
                mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(
            L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                             torch.arange(H)[None, :, None],
                             index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask


class CosBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, baseline=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        # self.attn = CosformerAttention(embed_dim=dim, num_heads=num_heads, causal=False).cuda().left_product
        # self.attn = Attention_(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # self.attn = LambdaLayer(1, num_heads=1, qk_ratio=1.0,linear="lambda")
        self.attn = InformerEncoderLayer(
            AttentionLayer(ProbAttention(False, factor=5, attention_dropout=attn_drop, output_attention=False),
                           d_model=dim, n_heads=num_heads, mix=False),
            dim,
            16,
            dropout=attn_drop,
            activation="gelu"
        )

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        if isinstance(self.attn,LambdaLayer):
            x = x + self.drop_path(self.attn(self.norm1(x).unsqueeze(1)).squeeze(1))
        elif isinstance(self.attn,InformerEncoderLayer):
            z = self.norm1(x)
            x = x + self.drop_path(self.attn(z,z))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class SeqentialMaskedAutoencoderConcatVersion(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    # 512の場合はここ変える

    # todo: in_chansを変える
    def __init__(self, img_size=256, patch_size=16, in_chans=1, 
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, baseline="attn", mask_token_type="all", mask_ratio=0.75): # all, sub
        super().__init__()
        assert baseline == "attn"
        k = in_chans
        in_chans = 1

        self.patch_size = patch_size
        self.practical_image_size = (img_size*k,img_size)
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(
            self.practical_image_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.embed_dim = embed_dim
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(
                1,
                num_patches,
                embed_dim),
            requires_grad=False) 

        self.blocks = nn.ModuleList([
            CosBlock(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
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
                num_patches if mask_token_type == "all" else num_patches - int((k-1) * (num_patches // k) * mask_ratio),
                decoder_embed_dim),
            requires_grad=False) 

        self.decoder_blocks = nn.ModuleList([
                CosBlock(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer) for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            patch_size**2 * in_chans,
            bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss
        self.mask_token_type = mask_token_type
        self.mask_ratio = mask_ratio

        self.initialize_weights(k)


    def initialize_weights(self,k):
        H, W = self.practical_image_size
        H, W = H // self.patch_size, W // self.patch_size

        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed_ex(self.pos_embed.shape[-1], H, W, cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        if self.mask_token_type == "all":
            decoder_pos_embed = get_2d_sincos_pos_embed_ex(self.decoder_pos_embed.shape[-1], H, W, cls_token=False)
            self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        else:
            H = H - int((k-1) * (H // k) * self.mask_ratio)
            decoder_pos_embed = get_2d_sincos_pos_embed_ex(self.decoder_pos_embed.shape[-1], H, W, cls_token=False)
            self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as
        # cutoff is too big (2.)
        # torch.nn.init.normal_(self.cls_token, std=.02)
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
        p = self.patch_embed.patch_size[0]
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
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 1, h * p, h * p))
        return imgs

    def calc_patch_distribution(self, x): # x: (B,C,H,W)
        x = self.patchify(x) # (B,L,D)
        x = torch.std(x, dim=2) # (B,L)
        means, stds = torch.mean(x,dim=1), torch.std(x,dim=1)
        return x, means, stds

    def get_masking_idx(self,x,original_idx,mask_ratio):
        L,D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(L, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        # ascend: small is keep, large is remove
        ids_shuffle = torch.argsort(noise, dim=0)
        ids_restore = torch.argsort(ids_shuffle, dim=0)
        ids_keep, ids_mask = ids_shuffle[:len_keep], ids_shuffle[len_keep:]
        return original_idx[ids_keep], original_idx[ids_restore], original_idx[ids_shuffle]
    
    def random_masking_vit(self, x, hd, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """

        debug = False
        algorithm = "top-u"

        if algorithm == "top-u":
            B, L, D = x.shape
            p_std, means, stds = self.calc_patch_distribution(hd)
            thr = stds * 0.75
            thr = thr.unsqueeze(1).repeat(1,L) # (B,L)
            means = means.unsqueeze(1).repeat(1,L) # (B,L)

            u = int(L * 0.1)
            top_idx = torch.argsort(p_std,dim=1,descending=True)[:, :u]
            index_b = top_idx
        
            pa, pb = torch.abs(p_std - means) < thr, torch.abs(p_std - means) >= thr # (B,L)
            pa, pb = pa.float(), pb.float()
            la = int(pa.sum(dim=1).mean())
            sampling_count_a = int(L * (1-mask_ratio)) - u
            index_a = pa.multinomial(num_samples=sampling_count_a, replacement=False) # (B,La)
            ids_keep = torch.cat([index_a,index_b],dim=1) # (B,L_keep) 
            ids_keep = torch.unique(ids_keep,dim=1)

            # index_aとindex_bが重複を含む
            ids_discard = torch.Tensor([]) # ids_keepの補集合
            for i in range(B):
                rest = torch.linspace(0,L-1,L)
                mk = torch.ones(L,dtype=torch.bool)
                mk[ids_keep[i]] = False
                ideal = L - ids_keep.shape[-1]
                offset = max(0,torch.sum(mk) - ideal) # ids_keepが少ない場合, offset分restを削る
                rest = rest[mk].unsqueeze(0)
                if offset > 0:
                    rest = rest[:,:-offset]
                ids_discard = torch.cat([ids_discard,rest],dim=0)

            # print(sampling_count,sampling_count_a,sampling_count_b,la,int(pb.sum(dim=1).mean()), ids_discard.shape)
            x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
            ids_shuffle = torch.cat([ids_keep,ids_discard.cuda()],dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)

            # generate the binary mask: 0 is keep, 1 is remove
            mask = torch.ones((B,L), device=x.device)

            if not debug:
                mask[:, :ids_keep.shape[-1]] = 0
            
            #####  group_b (stdでかい)に指定されているものだけmaskしない / mask=1 はマスクされるの意なので注意
            if debug:
                print("index_{a,b}.shape", index_a.shape[-1],index_b.shape[-1])
                mask[:,index_a.shape[-1]:index_a.shape[-1]+index_b.shape[-1]] = 0
            ####

            # unshuffle to get the binary mask
            mask = torch.gather(mask, dim=1, index=ids_restore)

            # print("x_masked",x_masked.shape)
            return x_masked, mask, ids_restore, ids_keep
        
        elif algorithm == "thr":
            B, L, D = x.shape
            p_std, means, stds = self.calc_patch_distribution(hd)
            thr = stds * 0.75
            thr = thr.unsqueeze(1).repeat(1,L) # (B,L)
            means = means.unsqueeze(1).repeat(1,L) # (B,L)
        
            pa, pb = torch.abs(p_std - means) < thr, torch.abs(p_std - means) >= thr # (B,L)
            pa, pb = pa.float(), pb.float()
            la = int(pa.sum(dim=1).mean())
            sampling_count = int(L * (1 - mask_ratio))
            sampling_count_a = int(la * (1 - mask_ratio))
            sampling_count_b = sampling_count - sampling_count_a
            index_a = pa.multinomial(num_samples=sampling_count_a, replacement=False) # (B,La)
            index_b = pb.multinomial(num_samples=sampling_count_b, replacement=False) # (B,Lb)
            ids_keep = torch.cat([index_a,index_b],dim=1) # (B,L_keep) 
            ids_keep = torch.unique(ids_keep,dim=1)

            # index_aとindex_bが重複を含む
            ids_discard = torch.Tensor([]) # ids_keepの補集合
            for i in range(B):
                rest = torch.linspace(0,L-1,L)
                mk = torch.ones(L,dtype=torch.bool)
                mk[ids_keep[i]] = False
                ideal = L - ids_keep.shape[-1]
                offset = max(0,torch.sum(mk) - ideal) # ids_keepが少ない場合, offset分restを削る
                rest = rest[mk].unsqueeze(0)
                if offset > 0:
                    rest = rest[:,:-offset]
                ids_discard = torch.cat([ids_discard,rest],dim=0)

            # print(sampling_count,sampling_count_a,sampling_count_b,la,int(pb.sum(dim=1).mean()), ids_discard.shape)
            x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
            ids_shuffle = torch.cat([ids_keep,ids_discard.cuda()],dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)

            # generate the binary mask: 0 is keep, 1 is remove
            mask = torch.ones((B,L), device=x.device)

            if not debug:
                mask[:, :ids_keep.shape[-1]] = 0
            
            #####  group_b (stdでかい)に指定されているものだけmaskしない / mask=1 はマスクされるの意なので注意
            if debug:
                for i in range(B):
                    mask[i,pb[i].bool()] = 0
            ####


            # unshuffle to get the binary mask
            mask = torch.gather(mask, dim=1, index=ids_restore)

            # print("x_masked",x_masked.shape)
            return x_masked, mask, ids_restore, ids_keep

        elif algorithm == "original":
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

            return x_masked, mask, ids_restore, ids_shuffle


    def forward_encoder(self,x,k,mask_ratio=None): # x: (B,K,C,H,W)
        # embed patches
        hd = x[:,0,:,:,:].clone()
        x = x.transpose(1,2).flatten(2,3) # (B,C,H*K,W)
        x = self.patch_embed(x) # (B,H*W/patch**2,embed_dim)
        
        # add pos embed w/o cls token
        x = x + self.pos_embed

        # masking: length -> length * mask_ratio
        if mask_ratio is not None:
            B, L, D = x.shape
            unit = L//k
            h = x[:,:unit,:].clone()
            h, mask, ids_restore, ids_keep = self.random_masking_vit(h, hd, mask_ratio)
            
            _ids_keep = ids_keep.clone()
            _ids_restore = ids_restore.clone()

            if self.mask_token_type == "all":
                for i in range(k-1):
                    offset = unit * (i+1)
                    _ids_keep = torch.cat([_ids_keep, offset + ids_keep.clone()],dim=1)
                    _ids_restore = torch.cat([_ids_restore, offset + ids_restore.clone()],dim=1)

                ids_keep = _ids_keep
                ids_restore = _ids_restore
            elif self.mask_token_type == "sub":
                for i in range(k-1):
                    offset = unit * (i+1)
                    _ids_keep = torch.cat([_ids_keep, offset + ids_keep.clone()],dim=1)

                ids_keep = _ids_keep

            x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.shape[-1]))
            
        else:
            mask, ids_restore, ids_shuffle = None, None, None


        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return x, mask, ids_restore, ids_keep
     
    def forward_decoder(self, x, k, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        if self.mask_token_type == "all":
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
            x_ = torch.cat([x, mask_tokens], dim=1)
            x = torch.gather(x_,
                            dim=1,
                            index=ids_restore.unsqueeze(-1).repeat(1,
                                                                    1,
                                                                    x.shape[2]))  # unshuffle
            
        elif self.mask_token_type == "sub":
            B,L,D = x.shape
            assert L % k == 0
            x, X = x[:,:L//k,:], x[:,L//k:,:]
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
            x_ = torch.cat([x, mask_tokens], dim=1)
            x = torch.gather(x_,
                            dim=1,
                            index=ids_restore.unsqueeze(-1).repeat(1,
                                                                    1,
                                                                    x.shape[2]))  # unshuffle
            x = torch.cat([x,X],dim=1)
            

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)

        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        return x

    def forward_loss(self, k, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        B, L, D = pred.shape
        if self.mask_token_type == "all":
            assert L % k == 0
            pred = pred[:,:L//k,:]
        else:
            prac_L = self.patch_embed.num_patches
            assert prac_L % k == 0 and prac_L - int((k-1) * (prac_L // k) * self.mask_ratio) == L
            pred = pred[:,:prac_L//k,:]

        target = imgs[:,0,:,:,:] # (B,C,H,W)
        target = self.patchify(target) # todo: (B,256,256) 
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        # print(loss.item())

        return loss, pred

    def forward(self, imgs,mask_ratio):
        _, k, _, _, _ = imgs.shape
        y, mask, ids_restore, _ = self.forward_encoder(imgs,k,mask_ratio=mask_ratio)
        pred = self.forward_decoder(y, k, ids_restore)  # [N, L, p*p*3]
        loss, pred = self.forward_loss(k, imgs, pred, mask)
        return loss, pred, mask


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_for_FT512d8b(embed_dim=512,**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=8, embed_dim=embed_dim, num_heads=8, # embed_dim % num_heads == 0 にしないとだめなので注意
        decoder_embed_dim=512, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
 

def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks



vit_for_FT = vit_for_FT512d8b  # decoder: 512 dim, 8 blocks, latent dim = 128
