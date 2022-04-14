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
from lib2to3.pgen2 import token
 
import torch
import torch.nn as nn

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



class TokenEmbed(nn.Module):
    def __init__(self, dim, window, token_window=2, embed_dim=768, norm_layer=None):
        assert window % token_window == 0

        super().__init__()
        self.dim = dim
        self.window = window
        self.token_window = token_window
        self.token_num = dim * window // token_window
        
        self.mlp = nn.Linear(token_window,embed_dim)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B,W,D = x.shape
        x = self.tokenize(x) 

        x = self.mlp(x)
        x = self.norm(x)
        return x

    def tokenize(self,x: torch.Tensor): # x: (B,W,D) -> (B,L,D)
        B,W,D = x.shape
        x = x.transpose(-1,-2).contiguous().view(B,-1,self.token_window) # (B,token_num,token_window) todo: ここチェック
        return x

    def detokenize(self,x:torch.Tensor):
        B,W,D = x.shape
        x = x.view(B,-1,self.dim).transpose(-1,-2)
        return x

        


class OneDimMaskedAutoencoder(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, dim, window, token_window=2,
                 embed_dim=90, depth=12, num_heads=16,
                 decoder_embed_dim=128, decoder_depth=8, decoder_num_heads=4,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, baseline="attn"):
        super().__init__()

        while embed_dim % 4 != 0:
            embed_dim += 1

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.token_embed = TokenEmbed(dim=dim, window=window, token_window=token_window, embed_dim=embed_dim)
        num_patches = self.token_embed.token_num

        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(
                1,
                num_patches,
                embed_dim),
            requires_grad=False)  # fixed sin-cos embedding

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
                num_patches,
                decoder_embed_dim),
            requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    baseline=baseline) for i in range(decoder_depth)]) # todo

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            token_window,
            bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss
        self.embed_dim = embed_dim
        self.window = window
        self.token_window = token_window
        self.dim = dim

        self.initialize_weights()


    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        print(self.token_embed.token_num,  self.window // self.token_window, self.dim, self.pos_embed.data.shape)
        pos_embed = get_2d_sincos_pos_embed_ex(self.pos_embed.shape[-1], self.window // self.token_window, self.dim, cls_token=False)

        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0)) # todo: ここチェック → 多分OK

        decoder_pos_embed = get_2d_sincos_pos_embed_ex(self.decoder_pos_embed.shape[-1], self.window // self.token_window, self.dim, cls_token=False)
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))  # todo: ここチェック

        # initialize token_embed like nn.Linear (instead of nn.Conv2d)
        # w = self.token_embed.proj.weight.data
        # torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

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

    def random_masking(self, x, mask_ratio):
        B,L,D = x.shape

        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(B, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        # ascend: small is keep, large is remove
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self,x,mask_ratio):
        # embed patches
        x = self.token_embed(x) # (B,L,D)

        # add pos embed w/o cls token
        x = x + self.pos_embed

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        # cls_token = self.cls_token + self.pos_embed[:, :1, :]
        # cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        # x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_,
                          dim=1,
                          index=ids_restore.unsqueeze(-1).repeat(1,
                                                                 1,
                                                                 x.shape[2]))  # unshuffle
        x = x_

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        # print("decoder",x.shape)
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        return x

    def forward_loss(self, x, pred, mask):
        """
        x: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.token_embed.tokenize(x)
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