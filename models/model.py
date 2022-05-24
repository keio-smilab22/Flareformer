import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attn import ProbAttention, AttentionLayer
from timm.models.layers import trunc_normal_, DropPath

class FlareFormer(nn.Module):
    def __init__(self, input_channel, output_channel, sfm_params, mm_params, window=24):
        super(FlareFormer, self).__init__()

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
        x = torch.cat((feat_output, img_output), 1)
        output = self.generator(x)

        output = self.softmax(output)

        return output, x 

    def freeze_feature_extractor(self):
        for param in self.parameters():
            param.requires_grad = False # 重み固定
        
        for param in self.generator.parameters():
            param.requires_grad = True




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




class PositionwiseFeedForward(torch.nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(d_model, d_ff)
        self.w_2 = torch.nn.Linear(d_ff, d_model)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

