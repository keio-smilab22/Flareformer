import torch
import torch.nn as nn
import torch.nn.functional as F

# from utils.masking import TriangularCausalMask, ProbMask
# from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
# from models.decoder import Decoder, DecoderLayer
# from models.attn import FullAttention, ProbAttention, AttentionLayer
# from models.embed import DataEmbedding

from src.model import *

class FlareTransformerRegression(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=128, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', 
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0'), has_vit_head=True):
        super(FlareTransformerRegression, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # self.mae_encoder = mae_encoder
        self.magnetogram_feature_extractor = CNNModel(
            output_channel=24, pretrain=False) # NOTE output_channel is not used

        # physical feature
        self.linear_in_1 = torch.nn.Linear(
            enc_in, d_model)  # 79 -> 256
        self.bn1 = torch.nn.BatchNorm1d(seq_len) # 128
        self.relu = torch.nn.ReLU()

        # Encoding
        # self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model*2, embed, freq, dropout)


        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder
        # self.magnetogram_encoder = InformerEncoderLayer(
        #             AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
        #                         d_model, n_heads, mix=False),
        #             d_model,
        #             d_ff,
        #             dropout=dropout,
        #             activation=activation
        # )
        # self.sunspot_feature_encoder = InformerEncoderLayer(
        #             AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
        #                         d_model, n_heads, mix=False),
        #             d_model,
        #             d_ff,
        #             dropout=dropout,
        #             activation=activation
        # )

        self.sunspot_feature_encoder = Encoder(
            [
                InformerEncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                )
                for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.magnetogram_encoder = Encoder(
            [
                InformerEncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                )
                for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.generator_phys = nn.Linear(d_model*seq_len, d_model)
        self.generator_image = nn.Linear(d_model*seq_len, d_model)

        # Decoder
        self.decoder = Decoder(
            [
                InformerDecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model*2, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model*2, n_heads, mix=False),
                    d_model*2,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model*2)
        )

        self.linear1 = nn.Linear(seq_len, out_len)
        self.projection = nn.Linear(d_model*2, c_out, bias=True)
        
    def forward(self, x_enc, x_mag, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        feat = x_enc
        imgs = x_mag

        # vit_head = self.vit_head or nn.Identity()
        img_feat = []
        # mae_feat = []
        for i, img in enumerate(imgs):
            img_out = self.magnetogram_feature_extractor(img)
            # mae_out = vit_head(self.mae_encoder.encode(img))
            img_feat.append(img_out)
            # mae_feat.append(mae_out)
        img_feat = torch.stack(img_feat, dim=0)
        # mae_feat = torch.stack(mae_feat, dim=0)

        # img_feat = torch.cat([img_feat, mae_feat], dim=2)

        # physcal feat
        phys_feat = self.linear_in_1(feat)
        phys_feat = self.bn1(phys_feat)
        phys_feat = self.relu(phys_feat)

        # concat
        merged_feat = torch.cat([phys_feat, img_feat], dim=1)

        # SFM
        feat_output = self.sunspot_feature_encoder(phys_feat, merged_feat)

        # MM
        img_output = self.magnetogram_encoder(img_feat, merged_feat)  #

        # Late fusion
        enc_out = torch.cat([feat_output, img_output], dim=-1)
        
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        dec_out = self.projection(dec_out)

        # if self.output_attention:
        #     return dec_out[:,-self.pred_len:,:], attns
        # else:
        # print(f"self.pred_len: {self.pred_len}")
        return dec_out[:,-self.pred_len:,:] # [B, L, D]
        # return dec_out[:,-1:,:] # [B, L, D]


class FlareTransformerRegressionMAE(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=128, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', 
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0'), has_vit_head=True):
        super(FlareTransformerRegressionMAE, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        mae_encoder = MaskedAutoEncoder(baseline="attn", embed_dim=64)
        if has_vit_head:
            self.vit_head = nn.Linear(mae_encoder.dim,d_model) # todo: window分headを用意する？
            d_model += d_model
        else:
            self.vit_head = None
            d_model += mae_encoder.dim


        self.mae_encoder = mae_encoder
        
        self.magnetogram_feature_extractor = CNNModel(
            output_channel=24, pretrain=False) # NOTE output_channel is not used

        # physical feature
        self.linear_in_1 = torch.nn.Linear(
            enc_in, d_model)  # 79 -> 256
        self.bn1 = torch.nn.BatchNorm1d(seq_len) # 128
        self.relu = torch.nn.ReLU()

        # Encoding
        # self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model*2, embed, freq, dropout)


        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder
        # self.magnetogram_encoder = InformerEncoderLayer(
        #             AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
        #                         d_model, n_heads, mix=False),
        #             d_model,
        #             d_ff,
        #             dropout=dropout,
        #             activation=activation
        # )
        # self.sunspot_feature_encoder = InformerEncoderLayer(
        #             AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
        #                         d_model, n_heads, mix=False),
        #             d_model,
        #             d_ff,
        #             dropout=dropout,
        #             activation=activation
        # )

        self.sunspot_feature_encoder = Encoder(
            [
                InformerEncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                )
                for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.magnetogram_encoder = Encoder(
            [
                InformerEncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                )
                for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.generator_phys = nn.Linear(d_model*seq_len, d_model)
        self.generator_image = nn.Linear(d_model*seq_len, d_model)

        # Decoder
        self.decoder = Decoder(
            [
                InformerDecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model*2, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model*2, n_heads, mix=False),
                    d_model*2,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model*2)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)

        self.linear1 = nn.Linear(seq_len, out_len)
        self.projection = nn.Linear(d_model*2, c_out, bias=True)
        
    def forward(self, x_enc, x_mag, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        feat = x_enc
        imgs = x_mag

        vit_head = self.vit_head or nn.Identity()
        img_feat = []
        mae_feat = []
        for i, img in enumerate(imgs):
            img_out = self.magnetogram_feature_extractor(img)
            mae_out = vit_head(self.mae_encoder.encode(img))
            img_feat.append(img_out)
            mae_feat.append(mae_out)
        img_feat = torch.stack(img_feat, dim=0)
        mae_feat = torch.stack(mae_feat, dim=0)

        img_feat = torch.cat([img_feat, mae_feat], dim=2)

        # physcal feat
        phys_feat = self.linear_in_1(feat)
        phys_feat = self.bn1(phys_feat)
        phys_feat = self.relu(phys_feat)

        # concat
        merged_feat = torch.cat([phys_feat, img_feat], dim=1)

        # SFM
        feat_output = self.sunspot_feature_encoder(phys_feat, merged_feat)
        # feat_output = torch.flatten(feat_output, 1, 2)  # [bs, k*SFM_d_model]
        # feat_output = self.generator_phys(feat_output)  # [bs, SFM_d_model]

        # MM
        img_output = self.magnetogram_encoder(img_feat, merged_feat)  #
        # img_output = torch.flatten(img_output, 1, 2)  # [bs, k*SFM_d_model]
        # img_output = self.generator_image(img_output)  # [bs, MM_d_model]

        # Late fusion
        enc_out = torch.cat([feat_output, img_output], dim=-1)
        

        # enc_out = self.enc_embedding(x_enc, x_mark_enc)
        # enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        
        # enc_out = enc_out.transpose(1, 2)
        # print(f"enc_out: {enc_out.shape}")
        # enc_out = self.linear1(enc_out)
        # enc_out = enc_out.transpose(1, 2)
        # print(f"enc_out_new: {enc_out.shape}")
        # dec_out = self.projection(enc_out)
        dec_out = self.projection(dec_out)

        # print(dec_out.shape)
        
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        # if self.output_attention:
        #     return dec_out[:,-self.pred_len:,:], attns
        # else:
        # print(f"self.pred_len: {self.pred_len}")
        return dec_out[:,-self.pred_len:,:] # [B, L, D]
        # return dec_out[:,-1:,:] # [B, L, D]


class FlareTransformerRegressionLastLinear(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=128, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', 
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0'), has_vit_head=True):
        super(FlareTransformerRegressionLastLinear, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # mae_encoder = MaskedAutoEncoder(baseline="attn", embed_dim=64)
        # if has_vit_head:
        #     self.vit_head = nn.Linear(mae_encoder.dim,d_model) # todo: window分headを用意する？
        #     d_model += d_model
        #     # sfm_params["d_model"] += d_out
        # else:
        #     self.vit_head = None
        #     d_model += mae_encoder.dim
        #     # mm_params["d_model"] += mae_encoder.dim
        #     # sfm_params["d_model"] += mae_encoder.dim


        # self.mae_encoder = mae_encoder
        self.magnetogram_feature_extractor = CNNModel(
            output_channel=24, pretrain=False) # NOTE output_channel is not used

        # physical feature
        self.linear_in_1 = torch.nn.Linear(
            enc_in, d_model)  # 79 -> 256
        self.bn1 = torch.nn.BatchNorm1d(seq_len) # 128
        self.relu = torch.nn.ReLU()

        # Encoding
        # self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model*2, embed, freq, dropout)


        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder
        # self.magnetogram_encoder = InformerEncoderLayer(
        #             AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
        #                         d_model, n_heads, mix=False),
        #             d_model,
        #             d_ff,
        #             dropout=dropout,
        #             activation=activation
        # )
        # self.sunspot_feature_encoder = InformerEncoderLayer(
        #             AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
        #                         d_model, n_heads, mix=False),
        #             d_model,
        #             d_ff,
        #             dropout=dropout,
        #             activation=activation
        # )

        self.sunspot_feature_encoder = Encoder(
            [
                InformerEncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                )
                for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.magnetogram_encoder = Encoder(
            [
                InformerEncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                )
                for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.generator_phys = nn.Linear(d_model*seq_len, d_model)
        self.generator_image = nn.Linear(d_model*seq_len, d_model)

        # Decoder
        self.decoder = Decoder(
            [
                InformerDecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model*2, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model*2, n_heads, mix=False),
                    d_model*2,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model*2)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)

        self.linear1 = nn.Linear(seq_len, out_len)
        self.projection = nn.Linear(d_model*2, c_out, bias=True)
        
    def forward(self, x_enc, x_mag, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        feat = x_enc
        imgs = x_mag

        # print("x_enc:", x_enc)
        # print("x_mag:", x_mag)
        # vit_head = self.vit_head or nn.Identity()
        img_feat = []
        # mae_feat = []
        for i, img in enumerate(imgs):
            img_out = self.magnetogram_feature_extractor(img)
            # mae_out = vit_head(self.mae_encoder.encode(img))
            img_feat.append(img_out)
            # mae_feat.append(mae_out)
        img_feat = torch.stack(img_feat, dim=0)
        # mae_feat = torch.stack(mae_feat, dim=0)

        # print(f"img_feat: {img_feat.shape}")
        # print(f"mae_feat: {mae_feat.shape}")    
        # img_feat = torch.cat([img_feat, mae_feat], dim=2)

        # physcal feat
        # print(feat.shape)
        phys_feat = self.linear_in_1(feat)
        phys_feat = self.bn1(phys_feat)
        phys_feat = self.relu(phys_feat)

        # concat
        merged_feat = torch.cat([phys_feat, img_feat], dim=1)

        # SFM
        feat_output = self.sunspot_feature_encoder(phys_feat, merged_feat)
        # feat_output = torch.flatten(feat_output, 1, 2)  # [bs, k*SFM_d_model]
        # feat_output = self.generator_phys(feat_output)  # [bs, SFM_d_model]

        # MM
        img_output = self.magnetogram_encoder(img_feat, merged_feat)  #
        # img_output = torch.flatten(img_output, 1, 2)  # [bs, k*SFM_d_model]
        # img_output = self.generator_image(img_output)  # [bs, MM_d_model]

        # Late fusion
        enc_out = torch.cat([feat_output, img_output], dim=-1)
        

        # enc_out = self.enc_embedding(x_enc, x_mark_enc)
        # enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # dec_out = self.dec_embedding(x_dec, x_mark_dec)
        # dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        
        enc_out = enc_out.transpose(1, 2)
        # print(f"enc_out: {enc_out.shape}")
        enc_out = self.linear1(enc_out)
        enc_out = enc_out.transpose(1, 2)
        # print(f"enc_out_new: {enc_out.shape}")
        dec_out = self.projection(enc_out)
        # dec_out = self.projection(dec_out)

        # print(dec_out.shape)
        
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        # if self.output_attention:
        #     return dec_out[:,-self.pred_len:,:], attns
        # else:
        # print(f"self.pred_len: {self.pred_len}")
        return dec_out[:,-self.pred_len:,:] # [B, L, D]
        # return dec_out[:,-1:,:] # [B, L, D]


class FlareTransformerRegressionrLastLinearMAE(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=128, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', 
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0'), has_vit_head=True):
        super(FlareTransformerRegressionrLastLinearMAE, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        mae_encoder = MaskedAutoEncoder(baseline="attn", embed_dim=64)
        if has_vit_head:
            self.vit_head = nn.Linear(mae_encoder.dim,d_model) # todo: window分headを用意する？
            d_model += d_model
            # sfm_params["d_model"] += d_out
        else:
            self.vit_head = None
            d_model += mae_encoder.dim
            # mm_params["d_model"] += mae_encoder.dim
            # sfm_params["d_model"] += mae_encoder.dim


        self.mae_encoder = mae_encoder
        self.magnetogram_feature_extractor = CNNModel(
            output_channel=24, pretrain=False) # NOTE output_channel is not used

        # physical feature
        self.linear_in_1 = torch.nn.Linear(
            enc_in, d_model)  # 79 -> 256
        self.bn1 = torch.nn.BatchNorm1d(seq_len) # 128
        self.relu = torch.nn.ReLU()

        # Encoding
        # self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model*2, embed, freq, dropout)


        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder
        # self.magnetogram_encoder = InformerEncoderLayer(
        #             AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
        #                         d_model, n_heads, mix=False),
        #             d_model,
        #             d_ff,
        #             dropout=dropout,
        #             activation=activation
        # )
        # self.sunspot_feature_encoder = InformerEncoderLayer(
        #             AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
        #                         d_model, n_heads, mix=False),
        #             d_model,
        #             d_ff,
        #             dropout=dropout,
        #             activation=activation
        # )

        # self.sunspot_feature_encoder = Encoder(
        #     [
        #         InformerEncoderLayer(
        #             AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
        #                         d_model, n_heads, mix=False),
        #             d_model,
        #             d_ff,
        #             dropout=dropout,
        #             activation=activation
        #         )
        #         for l in range(e_layers)
        #     ],
        #     norm_layer=torch.nn.LayerNorm(d_model)
        # )

        # self.magnetogram_encoder = Encoder(
        #     [
        #         InformerEncoderLayer(
        #             AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
        #                         d_model, n_heads, mix=False),
        #             d_model,
        #             d_ff,
        #             dropout=dropout,
        #             activation=activation
        #         )
        #         for l in range(e_layers)
        #     ],
        #     norm_layer=torch.nn.LayerNorm(d_model)
        # )

        self.sunspot_feature_encoder = Encoder(
            [
                InformerEncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                )
                for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.magnetogram_encoder = Encoder(
            [
                InformerEncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                )
                for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.generator_phys = nn.Linear(d_model*seq_len, d_model)
        self.generator_image = nn.Linear(d_model*seq_len, d_model)

        # Decoder
        self.decoder = Decoder(
            [
                InformerDecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model*2, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model*2, n_heads, mix=False),
                    d_model*2,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model*2)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)

        self.linear1 = nn.Linear(seq_len, out_len)
        self.projection = nn.Linear(d_model*2, c_out, bias=True)
        
    def forward(self, x_enc, x_mag, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        feat = x_enc
        imgs = x_mag

        # print("x_enc:", x_enc)
        # print("x_mag:", x_mag)
        vit_head = self.vit_head or nn.Identity()
        img_feat = []
        mae_feat = []
        for i, img in enumerate(imgs):
            img_out = self.magnetogram_feature_extractor(img)
            mae_out = vit_head(self.mae_encoder.encode(img))
            img_feat.append(img_out)
            mae_feat.append(mae_out)
        img_feat = torch.stack(img_feat, dim=0)
        mae_feat = torch.stack(mae_feat, dim=0)

        # print(f"img_feat: {img_feat.shape}")
        # print(f"mae_feat: {mae_feat.shape}")    
        img_feat = torch.cat([img_feat, mae_feat], dim=2)

        # physcal feat
        # print(feat.shape)
        phys_feat = self.linear_in_1(feat)
        phys_feat = self.bn1(phys_feat)
        phys_feat = self.relu(phys_feat)

        # concat
        merged_feat = torch.cat([phys_feat, img_feat], dim=1)

        # SFM
        feat_output = self.sunspot_feature_encoder(phys_feat, merged_feat)
        # feat_output = torch.flatten(feat_output, 1, 2)  # [bs, k*SFM_d_model]
        # feat_output = self.generator_phys(feat_output)  # [bs, SFM_d_model]

        # MM
        img_output = self.magnetogram_encoder(img_feat, merged_feat)  #
        # img_output = torch.flatten(img_output, 1, 2)  # [bs, k*SFM_d_model]
        # img_output = self.generator_image(img_output)  # [bs, MM_d_model]

        # Late fusion
        enc_out = torch.cat([feat_output, img_output], dim=-1)
        

        # enc_out = self.enc_embedding(x_enc, x_mark_enc)
        # enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # dec_out = self.dec_embedding(x_dec, x_mark_dec)
        # dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        
        enc_out = enc_out.transpose(1, 2)
        # print(f"enc_out: {enc_out.shape}")
        enc_out = self.linear1(enc_out)
        enc_out = enc_out.transpose(1, 2)
        # print(f"enc_out_new: {enc_out.shape}")
        dec_out = self.projection(enc_out)
        # dec_out = self.projection(dec_out)

        # print(dec_out.shape)
        
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        # if self.output_attention:
        #     return dec_out[:,-self.pred_len:,:], attns
        # else:
        # print(f"self.pred_len: {self.pred_len}")
        return dec_out[:,-self.pred_len:,:] # [B, L, D]


class FlareTransformerRegressionWithoutPhys(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=128, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', 
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0'), has_vit_head=True):
        super(FlareTransformerRegressionWithoutPhys, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # self.mae_encoder = mae_encoder
        self.magnetogram_feature_extractor = CNNModel(
            output_channel=24, pretrain=False) # NOTE output_channel is not used

        # physical feature
        self.linear_in_1 = torch.nn.Linear(
            enc_in, d_model)  # 79 -> 256
        self.bn1 = torch.nn.BatchNorm1d(seq_len) # 128
        self.relu = torch.nn.ReLU()

        # Encoding
        # self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)


        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder
        # self.magnetogram_encoder = InformerEncoderLayer(
        #             AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
        #                         d_model, n_heads, mix=False),
        #             d_model,
        #             d_ff,
        #             dropout=dropout,
        #             activation=activation
        # )
        # self.sunspot_feature_encoder = InformerEncoderLayer(
        #             AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
        #                         d_model, n_heads, mix=False),
        #             d_model,
        #             d_ff,
        #             dropout=dropout,
        #             activation=activation
        # )

        self.sunspot_feature_encoder = Encoder(
            [
                InformerEncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                )
                for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.magnetogram_encoder = Encoder(
            [
                InformerEncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                )
                for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.generator_phys = nn.Linear(d_model*seq_len, d_model)
        self.generator_image = nn.Linear(d_model*seq_len, d_model)

        # Decoder
        self.decoder = Decoder(
            [
                InformerDecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.linear1 = nn.Linear(seq_len, out_len)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        
    def forward(self, x_enc, x_mag, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        feat = x_enc
        imgs = x_mag

        # vit_head = self.vit_head or nn.Identity()
        img_feat = []
        # mae_feat = []
        for i, img in enumerate(imgs):
            img_out = self.magnetogram_feature_extractor(img)
            # mae_out = vit_head(self.mae_encoder.encode(img))
            img_feat.append(img_out)
            # mae_feat.append(mae_out)
        img_feat = torch.stack(img_feat, dim=0)
        # mae_feat = torch.stack(mae_feat, dim=0)

        # img_feat = torch.cat([img_feat, mae_feat], dim=2)

        # physcal feat
        # phys_feat = self.linear_in_1(feat)
        # phys_feat = self.bn1(phys_feat)
        # phys_feat = self.relu(phys_feat)

        # concat
        # merged_feat = torch.cat([phys_feat, img_feat], dim=1)

        # SFM
        # feat_output = self.sunspot_feature_encoder(phys_feat, merged_feat)

        # MM
        img_output = self.magnetogram_encoder(img_feat, img_feat)  #

        # Late fusion
        # enc_out = torch.cat([feat_output, img_output], dim=-1)
        enc_out = img_output
        

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        dec_out = self.projection(dec_out)

        # if self.output_attention:
        #     return dec_out[:,-self.pred_len:,:], attns
        # else:
        # print(f"self.pred_len: {self.pred_len}")
        return dec_out[:,-self.pred_len:,:] # [B, L, D]
        # return dec_out[:,-1:,:] # [B, L, D]


class FlareTransformerRegressionMAEWithoutPhys(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=128, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', 
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0'), has_vit_head=True):
        super(FlareTransformerRegressionMAEWithoutPhys, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        mae_encoder = MaskedAutoEncoder(baseline="attn", embed_dim=64)
        if has_vit_head:
            self.vit_head = nn.Linear(mae_encoder.dim,d_model) # todo: window分headを用意する？
            d_model += d_model
        else:
            self.vit_head = None
            d_model += mae_encoder.dim


        self.mae_encoder = mae_encoder
        
        self.magnetogram_feature_extractor = CNNModel(
            output_channel=24, pretrain=False) # NOTE output_channel is not used

        # physical feature
        self.linear_in_1 = torch.nn.Linear(
            enc_in, d_model)  # 79 -> 256
        self.bn1 = torch.nn.BatchNorm1d(seq_len) # 128
        self.relu = torch.nn.ReLU()

        # Encoding
        # self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)


        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder
        # self.magnetogram_encoder = InformerEncoderLayer(
        #             AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
        #                         d_model, n_heads, mix=False),
        #             d_model,
        #             d_ff,
        #             dropout=dropout,
        #             activation=activation
        # )

        # self.sunspot_feature_encoder = InformerEncoderLayer(
        #             AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
        #                         d_model, n_heads, mix=False),
        #             d_model,
        #             d_ff,
        #             dropout=dropout,
        #             activation=activation
        # )

        self.sunspot_feature_encoder = Encoder(
            [
                InformerEncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                )
                for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.magnetogram_encoder = Encoder(
            [
                InformerEncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                )
                for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.generator_phys = nn.Linear(d_model*seq_len, d_model)
        self.generator_image = nn.Linear(d_model*seq_len, d_model)

        # Decoder
        self.decoder = Decoder(
            [
                InformerDecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.linear1 = nn.Linear(seq_len, out_len)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        
    def forward(self, x_enc, x_mag, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        feat = x_enc
        imgs = x_mag

        vit_head = self.vit_head or nn.Identity()
        img_feat = []
        mae_feat = []
        for i, img in enumerate(imgs):
            img_out = self.magnetogram_feature_extractor(img)
            mae_out = vit_head(self.mae_encoder.encode(img))
            img_feat.append(img_out)
            mae_feat.append(mae_out)
        img_feat = torch.stack(img_feat, dim=0)
        mae_feat = torch.stack(mae_feat, dim=0)

        img_feat = torch.cat([img_feat, mae_feat], dim=2)

        # physcal feat
        # phys_feat = self.linear_in_1(feat)
        # phys_feat = self.bn1(phys_feat)
        # phys_feat = self.relu(phys_feat)

        # concat
        # merged_feat = torch.cat([phys_feat, img_feat], dim=1)

        # SFM
        # feat_output = self.sunspot_feature_encoder(phys_feat, merged_feat)

        # MM
        img_output = self.magnetogram_encoder(img_feat, img_feat)  #

        # Late fusion
        # enc_out = torch.cat([feat_output, img_output], dim=-1)
        enc_out = img_output
        

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        dec_out = self.projection(dec_out)

        # if self.output_attention:
        #     return dec_out[:,-self.pred_len:,:], attns
        # else:
        # print(f"self.pred_len: {self.pred_len}")
        return dec_out[:,-self.pred_len:,:] # [B, L, D]
        # return dec_out[:,-1:,:] # [B, L, D]