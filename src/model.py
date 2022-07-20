import copy
from heapq import merge
import math
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor
from src.attn import FullAttention, ProbAttention, AttentionLayer
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from mae.prod.eval import MaskedAutoEncoder, PyramidMaskedAutoEncoder

# early fusion 
class FlareTransformer(nn.Module):
    def __init__(self, input_channel, output_channel, sfm_params, mm_params, window=24):
        super(FlareTransformer, self).__init__()

        # Informer``
        self.magnetogram_module = InformerEncoderLayer(
            AttentionLayer(ProbAttention(False, factor=5, attention_dropout=mm_params["dropout"], output_attention=False),
                           d_model=mm_params["d_model"], n_heads=mm_params["h"], mix=False),
            mm_params["d_model"],
            mm_params["d_ff"],
            dropout=mm_params["dropout"],
            activation="relu"
        )

        self.sunspot_feature_module = InformerEncoderLayer(
            AttentionLayer(ProbAttention(False, factor=5, attention_dropout=sfm_params["dropout"], output_attention=False),
                           d_model=sfm_params["d_model"], n_heads=sfm_params["h"], mix=False),
            sfm_params["d_model"],
            sfm_params["d_ff"],
            dropout=sfm_params["dropout"],
            activation="relu"
        )

        # Image Feature Extractor
        self.magnetogram_feature_extractor = CNNModel(
            output_channel=output_channel, pretrain=False)

        self.generator = nn.Linear(sfm_params["d_model"]+mm_params["d_model"],
                                   output_channel)

        self.linear = nn.Linear(
            window*mm_params["d_model"]*2, sfm_params["d_model"])
        self.softmax = nn.Softmax(dim=1)

        self.generator_image = nn.Linear(
            mm_params["d_model"]*window, mm_params["d_model"])

        self.generator_phys = nn.Linear(
            sfm_params["d_model"]*window, sfm_params["d_model"])
        self.relu = torch.nn.ReLU()
        self.linear_in_1 = torch.nn.Linear(
            input_channel, sfm_params["d_model"])  # 79 -> 128
        self.bn1 = torch.nn.BatchNorm1d(window)  # 128

    def forward_mm_feature_extractor(self, img_list, feat):
        for i, img in enumerate(img_list):  # img_list = [bs, k, 256]
            img_output = self.magnetogram_feature_extractor(img)
            if i == 0:
                img_feat = img_output.unsqueeze(0)
            else:
                img_feat = torch.cat(
                    [img_feat, img_output.unsqueeze(0)], dim=0)

        return img_feat

    def forward_sfm_feature_extractor(self, img_list, feat):
        phys_feat = self.linear_in_1(feat)
        phys_feat = self.bn1(phys_feat)
        phys_feat = self.relu(phys_feat)
        return phys_feat


    def forward(self, img_list, feat):
        # img_feat[bs, k, mm_d_model]
        img_feat = self.forward_mm_feature_extractor(img_list,feat)

        # physical feat
        phys_feat = self.forward_sfm_feature_extractor(img_list,feat)

        # concat
        merged_feat = torch.cat([phys_feat, img_feat], dim=1)

        # SFM
        feat_output = self.sunspot_feature_module(phys_feat, merged_feat)  #
        feat_output = torch.flatten(feat_output, 1, 2)  # [bs, k*SFM_d_model]
        feat_output = self.generator_phys(feat_output)  # [bs, SFM_d_model]

        # MM
        img_output = self.magnetogram_module(img_feat, merged_feat)  #
        img_output = torch.flatten(img_output, 1, 2)  # [bs, k*SFM_d_model]
        img_output = self.generator_image(img_output)  # [bs, MM_d_model]

        # Late fusion
        output = torch.cat((feat_output, img_output), 1)
        output = self.generator(output)

        output = self.softmax(output)

        return output


class FlareTransformerWithoutMM(nn.Module): # SFM only のFT
    def __init__(self, input_channel, output_channel, sfm_params, mm_params, window=24):
        super(FlareTransformerWithoutMM, self).__init__()
        self.sfm_params = sfm_params

        # Informer
        self.sunspot_feature_module = InformerEncoderLayer(
            AttentionLayer(ProbAttention(False, factor=5, attention_dropout=sfm_params["dropout"], output_attention=False),
                           d_model=sfm_params["d_model"], n_heads=sfm_params["h"], mix=False),
            sfm_params["d_model"],
            sfm_params["d_ff"],
            dropout=sfm_params["dropout"],
            activation="relu"
        )

        # Image Feature Extractor
        self.magnetogram_feature_extractor = CNNModel(
            output_channel=output_channel, pretrain=False)

        self.generator = nn.Linear(sfm_params["d_model"],
                                   output_channel)

        self.linear = nn.Linear(
            window*mm_params["d_model"]*2, sfm_params["d_model"])
        self.softmax = nn.Softmax(dim=1)

        self.generator_phys = nn.Linear(
            sfm_params["d_model"]*window, sfm_params["d_model"])
        self.relu = torch.nn.ReLU()
        self.linear_in_1 = torch.nn.Linear(
            input_channel, sfm_params["d_model"])  # 79 -> 128
        self.bn1 = torch.nn.BatchNorm1d(window)  # 128

    def forward(self, img_list, feat):
        # physical feat
        phys_feat = self.linear_in_1(feat)
        phys_feat = self.bn1(phys_feat)
        phys_feat = self.relu(phys_feat)

        # SFM
        for _ in range(self.sfm_params["N"]):
            phys_feat = self.sunspot_feature_module(phys_feat, phys_feat)  # todo: ここ同じモジュール使ったらあかんわ

        feat_output = torch.flatten(phys_feat, 1, 2)  # [bs, k*SFM_d_model]
        feat_output = self.generator_phys(feat_output)  # [bs, SFM_d_model]

        # Late fusion
        output = self.generator(feat_output)
        output = self.softmax(output)

        return output


class PureTransformerSFM(nn.Module): # Vanilla Transformerで書き換えたFT
    def __init__(self, input_channel, output_channel, sfm_params, mm_params, window=24):
        super(PureTransformerSFM, self).__init__()
        self.sfm_params = sfm_params

        d_hid = 200
        nlayers = self.sfm_params["N"]
        encoder_layers = TransformerEncoderLayer(sfm_params["d_model"], sfm_params["h"], d_hid, sfm_params["dropout"])
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        # Informer
        self.sunspot_feature_module = InformerEncoderLayer(
            AttentionLayer(ProbAttention(False, factor=5, attention_dropout=sfm_params["dropout"], output_attention=False),
                           d_model=sfm_params["d_model"], n_heads=sfm_params["h"], mix=False),
            sfm_params["d_model"],
            sfm_params["d_ff"],
            dropout=sfm_params["dropout"],
            activation="relu"
        )

        # Image Feature Extractor
        self.magnetogram_feature_extractor = CNNModel(
            output_channel=output_channel, pretrain=False)

        self.generator = nn.Linear(sfm_params["d_model"],
                                   output_channel)

        self.linear = nn.Linear(
            window*mm_params["d_model"]*2, sfm_params["d_model"])
        self.softmax = nn.Softmax(dim=1)

        self.generator_phys = nn.Linear(
            sfm_params["d_model"]*window, sfm_params["d_model"])
        self.relu = torch.nn.ReLU()
        self.linear_in_1 = torch.nn.Linear(
            input_channel, sfm_params["d_model"])  # 79 -> 128
        self.bn1 = torch.nn.BatchNorm1d(window)  # 128

    def forward(self, img_list, feat):
        # physical feat
        phys_feat = self.linear_in_1(feat)
        phys_feat = self.bn1(phys_feat)
        phys_feat = self.relu(phys_feat)

        # SFM
        # for _ in range(self.sfm_params["N"]):
        #     phys_feat = self.sunspot_feature_module(phys_feat, phys_feat)  #

        phys_feat = self.transformer_encoder(phys_feat)

        feat_output = torch.flatten(phys_feat, 1, 2)  # [bs, k*SFM_d_model]
        feat_output = self.generator_phys(feat_output)  # [bs, SFM_d_model]

        # Late fusion
        output = self.generator(feat_output)
        output = self.softmax(output)

        return output


class FlareTransformerWithPositonalEncoding(nn.Module): # PEを加えたFT
    def __init__(self, input_channel, output_channel, sfm_params, mm_params, window=24):
        super(FlareTransformerWithPositonalEncoding, self).__init__()

        self.mm_pos_encoder = PositionalEncoding(mm_params["d_model"], mm_params["dropout"])
        self.sfm_pos_encoder = PositionalEncoding(sfm_params["d_model"], sfm_params["dropout"])

        magnetogram_module = []
        sunspot_feature_module = []

        # Informer
        for _ in range(mm_params["N"]):
            encoder = InformerEncoderLayer(
                AttentionLayer(ProbAttention(False, factor=5, attention_dropout=mm_params["dropout"], output_attention=False),
                            d_model=mm_params["d_model"], n_heads=mm_params["h"], mix=False),
                mm_params["d_model"],
                mm_params["d_ff"],
                dropout=mm_params["dropout"],
                activation="relu"
            )
            magnetogram_module.append(encoder)

        for _ in range(sfm_params["N"]):
            encoder = InformerEncoderLayer(
                AttentionLayer(ProbAttention(False, factor=5, attention_dropout=sfm_params["dropout"], output_attention=False),
                            d_model=sfm_params["d_model"], n_heads=sfm_params["h"], mix=False),
                sfm_params["d_model"],
                sfm_params["d_ff"],
                dropout=sfm_params["dropout"],
                activation="relu"
            )
            sunspot_feature_module.append(encoder)

        self.magnetogram_module = torch.nn.ModuleList(magnetogram_module)
        self.sunspot_feature_module = torch.nn.ModuleList(sunspot_feature_module)


        # Image Feature Extractor
        self.magnetogram_feature_extractor = CNNModel(
            output_channel=output_channel, pretrain=False)

        # print(sfm_params["d_model"],mm_params["d_model"],window)
        self.generator = nn.Linear(sfm_params["d_model"]*window+mm_params["d_model"]*window,
                                   output_channel)

        self.linear = nn.Linear(
            window*mm_params["d_model"]*2, sfm_params["d_model"])
        self.softmax = nn.Softmax(dim=1)

        self.generator_image = nn.Linear(
            mm_params["d_model"]*window, mm_params["d_model"])

        self.generator_phys = nn.Linear(
            sfm_params["d_model"]*window, sfm_params["d_model"])
        self.relu = torch.nn.ReLU()
        self.linear_in_1 = torch.nn.Linear(
            input_channel, sfm_params["d_model"])  # 79 -> 128
        self.bn1 = torch.nn.BatchNorm1d(window)  # 128
        self.mm_norm = nn.LayerNorm(mm_params["d_model"])
        self.sfm_norm = nn.LayerNorm(sfm_params["d_model"])

    def forward(self, img_list, feat):
        # img_feat[bs, k, mm_d_model]
        for i, img in enumerate(img_list):  # img_list = [bs, k, 256]
            img_output = self.magnetogram_feature_extractor(img)
            if i == 0:
                img_feat = img_output.unsqueeze(0)
            else:
                img_feat = torch.cat(
                    [img_feat, img_output.unsqueeze(0)], dim=0)

        # physical feat
        phys_feat = self.linear_in_1(feat)
        phys_feat = self.bn1(phys_feat)
        phys_feat = self.relu(phys_feat)

        # positional encoding
        N_m, N_s = len(self.magnetogram_module),len(self.sunspot_feature_module)
        img_feat = self.mm_pos_encoder(img_feat)
        phys_feat = self.sfm_pos_encoder(phys_feat)

        for i in range(max(N_m,N_s)):
            # concat
            merged_feat = torch.cat([phys_feat, img_feat], dim=1)
            # print(phys_feat.shape,img_feat.shape,merged_feat.shape)
            if i < N_s:
                # SFM
                phys_feat = self.sunspot_feature_module[i](phys_feat, merged_feat)  #
                # phys_feat = self.generator_phys(phys_feat)  # [bs, SFM_d_model]
                # phys_feat = self.sfm_norm(phys_feat)
            if i < N_m:
                # MM
                img_feat = self.magnetogram_module[i](img_feat, merged_feat)  #
                # img_feat = self.generator_image(img_feat)  # [bs, MM_d_model]
                # img_feat = self.mm_norm(img_feat)

        phys_feat = torch.flatten(phys_feat, 1, 2)  # [bs, k*SFM_d_model]
        img_feat = torch.flatten(img_feat, 1, 2)  # [bs, k*SFM_d_model]

        # Late fusion
        output = torch.cat((phys_feat, img_feat), 1)
        # print(phys_feat.shape,img_feat.shape,merged_feat.shape)
        # print(output.shape)
        output = self.generator(output)

        output = self.softmax(output)

        return output


class FlareTransformerWithoutPE(nn.Module): # 複数層FT
    def __init__(self, input_channel, output_channel, sfm_params, mm_params, window=24):
        super(FlareTransformerWithoutPE, self).__init__()

        self.mm_pos_encoder = PositionalEncoding(mm_params["d_model"], mm_params["dropout"])
        self.sfm_pos_encoder = PositionalEncoding(sfm_params["d_model"], sfm_params["dropout"])

        magnetogram_module = []
        sunspot_feature_module = []

        # Informer
        for _ in range(mm_params["N"]):
            encoder = InformerEncoderLayer(
                AttentionLayer(ProbAttention(False, factor=5, attention_dropout=mm_params["dropout"], output_attention=False),
                            d_model=mm_params["d_model"], n_heads=mm_params["h"], mix=False),
                mm_params["d_model"],
                mm_params["d_ff"],
                dropout=mm_params["dropout"],
                activation="relu"
            )
            magnetogram_module.append(encoder)

        for _ in range(sfm_params["N"]):
            encoder = InformerEncoderLayer(
                AttentionLayer(ProbAttention(False, factor=5, attention_dropout=sfm_params["dropout"], output_attention=False),
                            d_model=sfm_params["d_model"], n_heads=sfm_params["h"], mix=False),
                sfm_params["d_model"],
                sfm_params["d_ff"],
                dropout=sfm_params["dropout"],
                activation="relu"
            )
            sunspot_feature_module.append(encoder)

        self.magnetogram_module = torch.nn.ModuleList(magnetogram_module)
        self.sunspot_feature_module = torch.nn.ModuleList(sunspot_feature_module)


        # Image Feature Extractor
        self.magnetogram_feature_extractor = CNNModel(
            output_channel=output_channel, pretrain=False)

        # print(sfm_params["d_model"],mm_params["d_model"],window)
        self.generator = nn.Linear(sfm_params["d_model"]*window+mm_params["d_model"]*window,
                                   output_channel)

        self.linear = nn.Linear(
            window*mm_params["d_model"]*2, sfm_params["d_model"])
        self.softmax = nn.Softmax(dim=1)

        self.generator_image = nn.Linear(
            mm_params["d_model"]*window, mm_params["d_model"])

        self.generator_phys = nn.Linear(
            sfm_params["d_model"]*window, sfm_params["d_model"])
        self.relu = torch.nn.ReLU()
        self.linear_in_1 = torch.nn.Linear(
            input_channel, sfm_params["d_model"])  # 79 -> 128
        self.bn1 = torch.nn.BatchNorm1d(window)  # 128
        self.mm_norm = nn.LayerNorm(mm_params["d_model"])
        self.sfm_norm = nn.LayerNorm(sfm_params["d_model"])

    def forward(self, img_list, feat):
        # img_feat[bs, k, mm_d_model]
        for i, img in enumerate(img_list):  # img_list = [bs, k, 256]
            img_output = self.magnetogram_feature_extractor(img)
            if i == 0:
                img_feat = img_output.unsqueeze(0)
            else:
                img_feat = torch.cat(
                    [img_feat, img_output.unsqueeze(0)], dim=0)

        # physical feat
        phys_feat = self.linear_in_1(feat)
        phys_feat = self.bn1(phys_feat)
        phys_feat = self.relu(phys_feat)

        # positional encoding
        N_m, N_s = len(self.magnetogram_module),len(self.sunspot_feature_module)
        # img_feat = self.mm_pos_encoder(img_feat)
        # phys_feat = self.sfm_pos_encoder(phys_feat)

        for i in range(max(N_m,N_s)):
            # concat
            merged_feat = torch.cat([phys_feat, img_feat], dim=1)
            # print(phys_feat.shape,img_feat.shape,merged_feat.shape)
            if i < N_s:
                # SFM
                phys_feat = self.sunspot_feature_module[i](phys_feat, merged_feat)  #
            if i < N_m:
                # MM
                img_feat = self.magnetogram_module[i](img_feat, merged_feat)  #

        phys_feat = torch.flatten(phys_feat, 1, 2)  # [bs, k*SFM_d_model]
        img_feat = torch.flatten(img_feat, 1, 2)  # [bs, k*SFM_d_model]

        # Late fusion
        output = torch.cat((phys_feat, img_feat), 1)
        # print(phys_feat.shape,img_feat.shape,merged_feat.shape)
        # print(output.shape)
        output = self.generator(output)

        output = self.softmax(output)

        return output


class FlareTransformerLikeViLBERT(nn.Module): # 複数層のtransformer層を持つFT (ViLBERT参考)
    def __init__(self, input_channel, output_channel, sfm_params, mm_params, window=24):
        super(FlareTransformerLikeViLBERT, self).__init__()

        self.mm_pos_encoder = PositionalEncoding(mm_params["d_model"], mm_params["dropout"])
        self.sfm_pos_encoder = PositionalEncoding(sfm_params["d_model"], sfm_params["dropout"])

        magnetogram_module = []
        sunspot_feature_module = []

        self.sfm_feature_extractor = InformerEncoderLayer(
                AttentionLayer(ProbAttention(False, factor=5, attention_dropout=sfm_params["dropout"], output_attention=False),
                            d_model=input_channel, n_heads=sfm_params["h"], mix=False),
                input_channel,
                sfm_params["d_ff"],
                dropout=sfm_params["dropout"],
                activation="relu"
            )

        # Informer
        for _ in range(mm_params["N"]):
            encoder = InformerEncoderLayer(
                AttentionLayer(ProbAttention(False, factor=5, attention_dropout=mm_params["dropout"], output_attention=False),
                            d_model=mm_params["d_model"], n_heads=mm_params["h"], mix=False),
                mm_params["d_model"],
                mm_params["d_ff"],
                dropout=mm_params["dropout"],
                activation="relu"
            )
            magnetogram_module.append(encoder)

        for _ in range(sfm_params["N"]):
            encoder = InformerEncoderLayer(
                AttentionLayer(ProbAttention(False, factor=5, attention_dropout=sfm_params["dropout"], output_attention=False),
                            d_model=sfm_params["d_model"], n_heads=sfm_params["h"], mix=False),
                sfm_params["d_model"],
                sfm_params["d_ff"],
                dropout=sfm_params["dropout"],
                activation="relu"
            )
            sunspot_feature_module.append(encoder)

        self.magnetogram_module = torch.nn.ModuleList(magnetogram_module)
        self.sunspot_feature_module = torch.nn.ModuleList(sunspot_feature_module)


        # Image Feature Extractor
        self.magnetogram_feature_extractor = CNNModel(
            output_channel=output_channel, pretrain=False)

        # print(sfm_params["d_model"],mm_params["d_model"],window)
        self.generator = nn.Linear(sfm_params["d_model"]*window+mm_params["d_model"]*window,
                                   output_channel)

        self.linear = nn.Linear(
            window*mm_params["d_model"]*2, sfm_params["d_model"])
        self.softmax = nn.Softmax(dim=1)

        self.generator_image = nn.Linear(
            mm_params["d_model"]*window, mm_params["d_model"])

        self.generator_phys = nn.Linear(
            sfm_params["d_model"]*window, sfm_params["d_model"])
        self.relu = torch.nn.ReLU()
        self.linear_in_1 = torch.nn.Linear(
            input_channel, sfm_params["d_model"])  # 79 -> 128
        self.bn1 = torch.nn.BatchNorm1d(window)  # 128
        self.mm_norm = nn.LayerNorm(mm_params["d_model"])
        self.sfm_norm = nn.LayerNorm(sfm_params["d_model"])

    def forward(self, img_list, feat):
        # img_feat[bs, k, mm_d_model]
        for i, img in enumerate(img_list):  # img_list = [bs, k, 256]
            img_output = self.magnetogram_feature_extractor(img)
            if i == 0:
                img_feat = img_output.unsqueeze(0)
            else:
                img_feat = torch.cat(
                    [img_feat, img_output.unsqueeze(0)], dim=0)

        # physical feat

        phys_feat = self.sfm_feature_extractor(feat,feat) # SFMの最初の部分にTransformerを加える
        phys_feat = self.linear_in_1(phys_feat)
        phys_feat = self.bn1(phys_feat)
        phys_feat = self.relu(phys_feat)

        # positional encoding
        N_m, N_s = len(self.magnetogram_module),len(self.sunspot_feature_module)
        # img_feat = self.mm_pos_encoder(img_feat)
        # phys_feat = self.sfm_pos_encoder(phys_feat)

        for i in range(max(N_m,N_s)):
            # concat
            merged_feat = torch.cat([phys_feat, img_feat], dim=1)
            # print(phys_feat.shape,img_feat.shape,merged_feat.shape)
            if i < N_s:
                # SFM
                phys_feat = self.sunspot_feature_module[i](phys_feat, merged_feat)  #
            if i < N_m:
                # MM
                img_feat = self.magnetogram_module[i](img_feat, merged_feat)  #

        phys_feat = torch.flatten(phys_feat, 1, 2)  # [bs, k*SFM_d_model]
        img_feat = torch.flatten(img_feat, 1, 2)  # [bs, k*SFM_d_model]

        # Late fusion
        output = torch.cat((phys_feat, img_feat), 1)
        # print(phys_feat.shape,img_feat.shape,merged_feat.shape)
        # print(output.shape)
        output = self.generator(output)

        output = self.softmax(output)

        return output

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
 


class FlareTransformerWithMAE(nn.Module):  # MAEを持つFT
    def __init__(self, input_channel, output_channel, sfm_params, mm_params, window=24, baseline="attn", embed_dim=512, has_vit_head=False):
        super(FlareTransformerWithMAE, self).__init__()
        mae_encoder = MaskedAutoEncoder(baseline=baseline, embed_dim=embed_dim)
        # mae_encoder = PyramidMaskedAutoEncoder(baseline=baseline, embed_dim=embed_dim)

        for param in mae_encoder.model.parameters():
            param.requires_grad = False
        # mae_encoder.model.requires_grad = False

        if has_vit_head:
            d_out = mm_params["d_model"]
            self.vit_head = nn.Linear(mae_encoder.dim,d_out) # todo: window分headを用意する？
            mm_params["d_model"] += d_out
            sfm_params["d_model"] += d_out
        else:
            self.vit_head = None
            mm_params["d_model"] += mae_encoder.dim
            sfm_params["d_model"] += mae_encoder.dim

        self.mae_encoder = mae_encoder
        
        # Informer
        self.magnetogram_module = InformerEncoderLayer(
            AttentionLayer(ProbAttention(False, factor=5, attention_dropout=mm_params["dropout"], output_attention=False),
                           d_model=mm_params["d_model"], n_heads=mm_params["h"], mix=False),
            mm_params["d_model"],
            mm_params["d_ff"],
            dropout=mm_params["dropout"],
            activation="relu"
        )

        self.sunspot_feature_module = InformerEncoderLayer(
            AttentionLayer(ProbAttention(False, factor=5, attention_dropout=sfm_params["dropout"], output_attention=False),
                           d_model=sfm_params["d_model"], n_heads=sfm_params["h"], mix=False),
            sfm_params["d_model"],
            sfm_params["d_ff"],
            dropout=sfm_params["dropout"],
            activation="relu"
        )

        # Image Feature Extractor
        self.magnetogram_feature_extractor = CNNModel(
            output_channel=output_channel, pretrain=False)


        self.generator = nn.Linear(sfm_params["d_model"]+mm_params["d_model"],
                                   output_channel)

        self.linear = nn.Linear(
            window*mm_params["d_model"]*2, sfm_params["d_model"])
        self.softmax = nn.Softmax(dim=1)

        self.generator_image = nn.Linear(
            mm_params["d_model"]*window, mm_params["d_model"])

        self.generator_phys = nn.Linear(
            sfm_params["d_model"]*window, sfm_params["d_model"])
        self.relu = torch.nn.ReLU()
        self.linear_in_1 = torch.nn.Linear(
            input_channel, sfm_params["d_model"])  # 79 -> 128
        self.bn1 = torch.nn.BatchNorm1d(window)  # 128


    def forward(self, img_list, feat): 
        vit_head = self.vit_head or nn.Identity()
        for i, img in enumerate(img_list):  # img_list = [bs, k, 256]
            img_output = self.magnetogram_feature_extractor(img).unsqueeze(0)
            vit_output = vit_head(self.mae_encoder.encode(img)).unsqueeze(0)
            if i == 0:
                img_feat = img_output
                mae_feat = vit_output
            else:
                img_feat = torch.cat([img_feat, img_output], dim=0)
                mae_feat = torch.cat([mae_feat, vit_output], dim=0)
         
        mae_feat = mae_feat.cuda()
        img_feat = torch.cat([img_feat, mae_feat], dim=2) 
        # img_feat = mae_feat

        # physical feat
        phys_feat = self.linear_in_1(feat)
        phys_feat = self.bn1(phys_feat)
        phys_feat = self.relu(phys_feat)

        # concat
        merged_feat = torch.cat([phys_feat, img_feat], dim=1)

        # SFM
        feat_output = self.sunspot_feature_module(phys_feat, merged_feat)  # todo: なんかここでエラーでるので修正する
        feat_output = torch.flatten(feat_output, 1, 2)  # [bs, k*SFM_d_model]
        feat_output = self.generator_phys(feat_output)  # [bs, SFM_d_model]

        # MM
        img_output = self.magnetogram_module(img_feat, merged_feat)  #
        img_output = torch.flatten(img_output, 1, 2)  # [bs, k*SFM_d_model]
        img_output = self.generator_image(img_output)  # [bs, MM_d_model]

        # Late fusion
        output = torch.cat((feat_output, img_output), 1)
        output = self.generator(output)

        output = self.softmax(output)

        return output

class FlareTransformerReplacedViTWithMAE(nn.Module): # MAEで事前学習させたViTで書き換えたFT
    def __init__(self, input_channel, output_channel, sfm_params, mm_params, window=24):
        super(FlareTransformerReplacedViTWithMAE, self).__init__()

        # mae_dim = 128
        # mm_params["d_model"] += mae_dim
        # sfm_params["d_model"] += mae_dim # こっちにも足しておかないとattentionの内積が計算できない

        # Informer
        self.magnetogram_module = InformerEncoderLayer(
            AttentionLayer(ProbAttention(False, factor=5, attention_dropout=mm_params["dropout"], output_attention=False),
                           d_model=mm_params["d_model"], n_heads=mm_params["h"], mix=False),
            mm_params["d_model"],
            mm_params["d_ff"],
            dropout=mm_params["dropout"],
            activation="relu"
        )

        self.sunspot_feature_module = InformerEncoderLayer(
            AttentionLayer(ProbAttention(False, factor=5, attention_dropout=sfm_params["dropout"], output_attention=False),
                           d_model=sfm_params["d_model"], n_heads=sfm_params["h"], mix=False),
            sfm_params["d_model"],
            sfm_params["d_ff"],
            dropout=sfm_params["dropout"],
            activation="relu"
        )

        # Image Feature Extractor
        # self.magnetogram_feature_extractor = CNNModel(
        #     output_channel=output_channel, pretrain=False)
        


        self.generator = nn.Linear(sfm_params["d_model"]+mm_params["d_model"],
                                   output_channel)

        self.linear = nn.Linear(
            window*mm_params["d_model"]*2, sfm_params["d_model"])
        self.softmax = nn.Softmax(dim=1)

        self.generator_image = nn.Linear(
            mm_params["d_model"]*window, mm_params["d_model"])

        self.generator_phys = nn.Linear(
            sfm_params["d_model"]*window, sfm_params["d_model"])
        self.relu = torch.nn.ReLU()
        self.linear_in_1 = torch.nn.Linear(
            input_channel, sfm_params["d_model"])  # 79 -> 128
        self.bn1 = torch.nn.BatchNorm1d(window)  # 128

        mae = MaskedAutoEncoder()
        self.mae_encoder = mae.get_model()
        self.mae_encoder.set_train_flag_encoeder()

    def forward(self, img_list, feat):
        # img_feat[bs, k, mm_d_model]
        for i, img in enumerate(img_list):  # img_list = [bs, k, 256]
            # import numpy as np
            # print(np.max(img.cpu().numpy()))
            # import cv2
            # x = np.empty((256,256,3))
            # for i in range(3): x[:,:,i] = img[0,:,:].cpu().numpy()
            # # print(result.shape)
            # cv2.namedWindow('window')
            # cv2.imshow('window', x)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # assert np.max(img.cpu().numpy()) <= 1
            latent, _, _ = self.mae_encoder.forward_encoder(img, 0)
            img_output = latent[:,0,:] # CLSトークンのみ使用
            # exit(0)
            if i == 0:
                img_feat = img_output.unsqueeze(0)
            else:
                img_feat = torch.cat(
                    [img_feat, img_output.unsqueeze(0)], dim=0)
         
        # mae_feat = mae_feat.cuda()
        # img_feat = torch.cat([img_feat, mae_feat], dim=2) # 1024 + 128

        # physical feat
        phys_feat = self.linear_in_1(feat)
        phys_feat = self.bn1(phys_feat)
        phys_feat = self.relu(phys_feat)

        # concat
        # print(img_feat.shape,phys_feat.shape)  # maeをconcatしたら: torch.Size([8, 4, 1152]) torch.Size([8, 4, 128])
        merged_feat = torch.cat([phys_feat, img_feat], dim=1)
        # print(img_feat.shape,phys_feat.shape,merged_feat.shape)

        # SFM
        feat_output = self.sunspot_feature_module(phys_feat, merged_feat)  # todo: なんかここでエラーでるので修正する
        feat_output = torch.flatten(feat_output, 1, 2)  # [bs, k*SFM_d_model]
        feat_output = self.generator_phys(feat_output)  # [bs, SFM_d_model]

        # MM
        img_output = self.magnetogram_module(img_feat, merged_feat)  #
        img_output = torch.flatten(img_output, 1, 2)  # [bs, k*SFM_d_model]
        img_output = self.generator_image(img_output)  # [bs, MM_d_model]

        # Late fusion
        output = torch.cat((feat_output, img_output), 1)
        output = self.generator(output)

        output = self.softmax(output)

        return output

class FlareTransformerViTWithMaeAndPositionalEncoding(nn.Module): # MAEで事前学習させたViTで書き換えたFT
    def __init__(self, input_channel, output_channel, sfm_params, mm_params, window=24):
        super(FlareTransformerViTWithMaeAndPositionalEncoding, self).__init__()

        # mae_dim = 128
        # mm_params["d_model"] += mae_dim
        # sfm_params["d_model"] += mae_dim # こっちにも足しておかないとattentionの内積が計算できない

       
        self.mm_pos_encoder = PositionalEncoding(mm_params["d_model"], mm_params["dropout"])
        self.sfm_pos_encoder = PositionalEncoding(sfm_params["d_model"], sfm_params["dropout"])

        # Informer
        self.magnetogram_module = InformerEncoderLayer(
            AttentionLayer(ProbAttention(False, factor=5, attention_dropout=mm_params["dropout"], output_attention=False),
                           d_model=mm_params["d_model"], n_heads=mm_params["h"], mix=False),
            mm_params["d_model"],
            mm_params["d_ff"],
            dropout=mm_params["dropout"],
            activation="relu"
        )

        self.sunspot_feature_module = InformerEncoderLayer(
            AttentionLayer(ProbAttention(False, factor=5, attention_dropout=sfm_params["dropout"], output_attention=False),
                           d_model=sfm_params["d_model"], n_heads=sfm_params["h"], mix=False),
            sfm_params["d_model"],
            sfm_params["d_ff"],
            dropout=sfm_params["dropout"],
            activation="relu"
        )

        # Image Feature Extractor
        self.magnetogram_feature_extractor = CNNModel(
            output_channel=output_channel, pretrain=False)
        


        self.generator = nn.Linear(sfm_params["d_model"]+mm_params["d_model"],
                                   output_channel)

        self.linear = nn.Linear(
            window*mm_params["d_model"]*2, sfm_params["d_model"])
        self.softmax = nn.Softmax(dim=1)

        self.generator_image = nn.Linear(
            mm_params["d_model"]*window, mm_params["d_model"])

        self.generator_phys = nn.Linear(
            sfm_params["d_model"]*window, sfm_params["d_model"])
        self.relu = torch.nn.ReLU()
        self.linear_in_1 = torch.nn.Linear(
            input_channel, sfm_params["d_model"])  # 79 -> 128
        self.bn1 = torch.nn.BatchNorm1d(window)  # 128

        mae = MaskedAutoEncoder()
        self.mae_encoder = mae.get_model()
        self.mae_encoder.set_train_flag_encoeder()

    def forward(self, img_list, feat):
        # img_feat[bs, k, mm_d_model]
        for i, img in enumerate(img_list):  # img_list = [bs, k, 256]
            latent, _, _ = self.mae_encoder.forward_encoder(img, 0)
            img_output = latent[:,0,:] # CLSトークンのみ使用
            # exit(0)
            if i == 0:
                img_feat = img_output.unsqueeze(0)
            else:
                img_feat = torch.cat(
                    [img_feat, img_output.unsqueeze(0)], dim=0)
         
        # mae_feat = mae_feat.cuda()
        # img_feat = torch.cat([img_feat, mae_feat], dim=2) # 1024 + 128

        # physical feat
        phys_feat = self.linear_in_1(feat)
        phys_feat = self.bn1(phys_feat)
        phys_feat = self.relu(phys_feat)

        img_feat = self.mm_pos_encoder(img_feat)
        phys_feat = self.sfm_pos_encoder(phys_feat)

        # concat
        # print(img_feat.shape,phys_feat.shape)  # maeをconcatしたら: torch.Size([8, 4, 1152]) torch.Size([8, 4, 128])
        merged_feat = torch.cat([phys_feat, img_feat], dim=1)
        # print(img_feat.shape,phys_feat.shape,merged_feat.shape)

        # SFM
        feat_output = self.sunspot_feature_module(phys_feat, merged_feat)  # todo: なんかここでエラーでるので修正する
        feat_output = torch.flatten(feat_output, 1, 2)  # [bs, k*SFM_d_model]
        feat_output = self.generator_phys(feat_output)  # [bs, SFM_d_model]

        # MM
        img_output = self.magnetogram_module(img_feat, merged_feat)  #
        img_output = torch.flatten(img_output, 1, 2)  # [bs, k*SFM_d_model]
        img_output = self.generator_image(img_output)  # [bs, MM_d_model]

        # Late fusion
        output = torch.cat((feat_output, img_output), 1)
        output = self.generator(output)

        output = self.softmax(output)

        return output




class FlareTransformerReplacedFreezeViTWithMAE(nn.Module):
    def __init__(self, input_channel, output_channel, sfm_params, mm_params, window=24):
        super(FlareTransformerReplacedFreezeViTWithMAE, self).__init__()

        # mae_dim = 128
        # mm_params["d_model"] += mae_dim
        # sfm_params["d_model"] += mae_dim # こっちにも足しておかないとattentionの内積が計算できない

        # Informer
        self.magnetogram_module = InformerEncoderLayer(
            AttentionLayer(ProbAttention(False, factor=5, attention_dropout=mm_params["dropout"], output_attention=False),
                           d_model=mm_params["d_model"], n_heads=mm_params["h"], mix=False),
            mm_params["d_model"],
            mm_params["d_ff"],
            dropout=mm_params["dropout"],
            activation="relu"
        )

        self.sunspot_feature_module = InformerEncoderLayer(
            AttentionLayer(ProbAttention(False, factor=5, attention_dropout=sfm_params["dropout"], output_attention=False),
                           d_model=sfm_params["d_model"], n_heads=sfm_params["h"], mix=False),
            sfm_params["d_model"],
            sfm_params["d_ff"],
            dropout=sfm_params["dropout"],
            activation="relu"
        )

        # Image Feature Extractor
        # self.magnetogram_feature_extractor = CNNModel(
        #     output_channel=output_channel, pretrain=False)
        


        self.generator = nn.Linear(sfm_params["d_model"]+mm_params["d_model"],
                                   output_channel)

        self.linear = nn.Linear(
            window*mm_params["d_model"]*2, sfm_params["d_model"])
        self.softmax = nn.Softmax(dim=1)

        self.generator_image = nn.Linear(
            mm_params["d_model"]*window, mm_params["d_model"])

        self.generator_phys = nn.Linear(
            sfm_params["d_model"]*window, sfm_params["d_model"])
        self.relu = torch.nn.ReLU()
        self.linear_in_1 = torch.nn.Linear(
            input_channel, sfm_params["d_model"])  # 79 -> 128
        self.bn1 = torch.nn.BatchNorm1d(window)  # 128

        mae = MaskedAutoEncoder()
        self.mae_encoder = mae.get_model()
        self.mae_encoder.set_train_flag_encoeder()
        for param in self.mae_encoder.parameters():
            param.requires_grad = False

    def forward(self, img_list, feat):
        # img_feat[bs, k, mm_d_model]
        for i, img in enumerate(img_list):  # img_list = [bs, k, 256]
            # import numpy as np
            # print(np.max(img.cpu().numpy()))
            # import cv2
            # x = np.empty((256,256,3))
            # for i in range(3): x[:,:,i] = img[0,:,:].cpu().numpy()
            # # print(result.shape)
            # cv2.namedWindow('window')
            # cv2.imshow('window', x)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # assert np.max(img.cpu().numpy()) <= 1
            latent, _, _ = self.mae_encoder.forward_encoder(img, 0)
            img_output = latent[:,0,:] # CLSトークンのみ使用
            # exit(0)
            if i == 0:
                img_feat = img_output.unsqueeze(0)
            else:
                img_feat = torch.cat(
                    [img_feat, img_output.unsqueeze(0)], dim=0)
         
        # mae_feat = mae_feat.cuda()
        # img_feat = torch.cat([img_feat, mae_feat], dim=2) # 1024 + 128

        # physical feat
        phys_feat = self.linear_in_1(feat)
        phys_feat = self.bn1(phys_feat)
        phys_feat = self.relu(phys_feat)

        # concat
        # print(img_feat.shape,phys_feat.shape)  # maeをconcatしたら: torch.Size([8, 4, 1152]) torch.Size([8, 4, 128])
        merged_feat = torch.cat([phys_feat, img_feat], dim=1)
        # print(img_feat.shape,phys_feat.shape,merged_feat.shape)

        # SFM
        feat_output = self.sunspot_feature_module(phys_feat, merged_feat)  # todo: なんかここでエラーでるので修正する
        feat_output = torch.flatten(feat_output, 1, 2)  # [bs, k*SFM_d_model]
        feat_output = self.generator_phys(feat_output)  # [bs, SFM_d_model]

        # MM
        img_output = self.magnetogram_module(img_feat, merged_feat)  #
        img_output = torch.flatten(img_output, 1, 2)  # [bs, k*SFM_d_model]
        img_output = self.generator_image(img_output)  # [bs, MM_d_model]

        # Late fusion
        output = torch.cat((feat_output, img_output), 1)
        output = self.generator(output)

        output = self.softmax(output)

        return output


from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm2(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, out_chans=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm2(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm2(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        # self.head = nn.Linear(dims[-1], num_classes)
        self.linear = nn.Linear(dims[-1], out_chans)

        self.apply(self._init_weights)
        # self.head.weight.data.mul_(head_init_scale)
        # self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.linear(x)
        # x = self.head(x)
        return x

class LayerNorm2(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class FlareTransformerWithConvNext(nn.Module):
    def __init__(self, input_channel, output_channel, sfm_params, mm_params, window=24):
        super(FlareTransformerWithConvNext, self).__init__()

        # Informer``
        self.magnetogram_module = InformerEncoderLayer(
            AttentionLayer(ProbAttention(False, factor=5, attention_dropout=mm_params["dropout"], output_attention=False),
                           d_model=mm_params["d_model"], n_heads=mm_params["h"], mix=False),
            mm_params["d_model"],
            mm_params["d_ff"],
            dropout=mm_params["dropout"],
            activation="relu"
        )

        self.sunspot_feature_module = InformerEncoderLayer(
            AttentionLayer(ProbAttention(False, factor=5, attention_dropout=sfm_params["dropout"], output_attention=False),
                           d_model=sfm_params["d_model"], n_heads=sfm_params["h"], mix=False),
            sfm_params["d_model"],
            sfm_params["d_ff"],
            dropout=sfm_params["dropout"],
            activation="relu"
        )

        # Image Feature Extractor
        # self.magnetogram_feature_extractor = CNNModel(
        #     output_channel=output_channel, pretrain=False)

        # resnet18 + ConvNext
        self.magnetogram_feature_extractor = ConvNeXt(in_chans=1,out_chans=mm_params["d_model"],depths=[2,2,2,2],dims=[64,128,256,512])

        self.generator = nn.Linear(sfm_params["d_model"]+mm_params["d_model"],
                                   output_channel)

        self.linear = nn.Linear(
            window*mm_params["d_model"]*2, sfm_params["d_model"])
        self.softmax = nn.Softmax(dim=1)

        self.generator_image = nn.Linear(
            mm_params["d_model"]*window, mm_params["d_model"])

        self.generator_phys = nn.Linear(
            sfm_params["d_model"]*window, sfm_params["d_model"])
        self.relu = torch.nn.ReLU()
        self.linear_in_1 = torch.nn.Linear(
            input_channel, sfm_params["d_model"])  # 79 -> 128
        self.bn1 = torch.nn.BatchNorm1d(window)  # 128

    def forward(self, img_list, feat):
        # img_feat[bs, k, mm_d_model]
        for i, img in enumerate(img_list):  # img_list = [bs, k, 256]
            img_output = self.magnetogram_feature_extractor(img)
            if i == 0:
                img_feat = img_output.unsqueeze(0)
            else:
                img_feat = torch.cat(
                    [img_feat, img_output.unsqueeze(0)], dim=0)

        # physical feat
        phys_feat = self.linear_in_1(feat)
        phys_feat = self.bn1(phys_feat)
        phys_feat = self.relu(phys_feat)

        # concat
        merged_feat = torch.cat([phys_feat, img_feat], dim=1)

        # SFM
        feat_output = self.sunspot_feature_module(phys_feat, merged_feat)  #
        feat_output = torch.flatten(feat_output, 1, 2)  # [bs, k*SFM_d_model]
        feat_output = self.generator_phys(feat_output)  # [bs, SFM_d_model]

        # MM
        img_output = self.magnetogram_module(img_feat, merged_feat)  #
        img_output = torch.flatten(img_output, 1, 2)  # [bs, k*SFM_d_model]
        img_output = self.generator_image(img_output)  # [bs, MM_d_model]

        # Late fusion
        output = torch.cat((feat_output, img_output), 1)
        output = self.generator(output)

        output = self.softmax(output)

        return output



class SunspotFeatureModule(torch.nn.Module):
    def __init__(self, N=6,
                 d_model=256, h=4, d_ff=16, dropout=0.1, mid_output=False, window=1):
        super(SunspotFeatureModule, self).__init__()
        self.mid_output = mid_output
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.encoder = Encoder(N, EncoderLayer(
            d_model, c(attn), c(ff), dropout=dropout))

        self.relu = torch.nn.ReLU()
        self.bn2 = torch.nn.BatchNorm1d(window*2)

    def forward(self, x):
        output = x
        output = self.encoder(output)  # [bs, 1, d_model]
        output = self.bn2(output)
        output = self.relu(output)

        return output  # [bs, d_model]


class Encoder(torch.nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, N, layer):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x



class _Encoder(torch.nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, N, layer):
        super(_Encoder, self).__init__()
        self.layers = clones(layer, N)
        # self.norm = LayerNorm(layer.size)

    def forward(self, q, kv):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(q, kv)
        # x = self.norm(x)
        return x


def clones(module, N):
    "Produce N identical layers."
    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class EncoderLayer(torch.nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x))
        return self.sublayer[1](x, self.feed_forward)



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

    def forward(self, q, attn_mask=None):
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
        kv = attn_mask
        new_x, attn = self.attention(
            q, kv, kv,
            attn_mask=None
        )
        q = q + self.dropout(new_x)

        y = q = self.norm1(q)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(q+y), attn


class InformerDecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(InformerDecoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        return self.norm3(x+y)

class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x

class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x#, attns


class SublayerConnection(torch.nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class LayerNorm(torch.nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = torch.nn.Parameter(torch.ones(features))
        self.b_2 = torch.nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class PositionwiseFeedForward(torch.nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(d_model, d_ff)
        self.w_2 = torch.nn.Linear(d_ff, d_model)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class MultiHeadedAttention(torch.nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(torch.nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
        / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class CNNModel(nn.Module):
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
        self.layer1 = Bottleneck(
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


class Bottleneck(nn.Module):
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
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


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


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, 
                                    kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
        return x

class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4; hour_size = 24
        weekday_size = 7; day_size = 32; month_size = 13

        Embed = FixedEmbedding if embed_type=='fixed' else nn.Embedding
        if freq=='t':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)
    
    def forward(self, x):
        x = x.long()
        
        minute_x = self.minute_embed(x[:,:,4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:,:,3])
        weekday_x = self.weekday_embed(x[:,:,2])
        day_x = self.day_embed(x[:,:,1])
        month_x = self.month_embed(x[:,:,0])
        
        return hour_x + weekday_x + day_x + month_x + minute_x

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h':4, 't':5, 's':6, 'm':1, 'a':1, 'w':2, 'd':3, 'b':3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)
    
    def forward(self, x):
        return self.embed(x)

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) if embed_type!='timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
        
        return self.dropout(x)
