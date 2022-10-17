import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attn import ProbAttention, AttentionLayer
from timm.models.layers import trunc_normal_, DropPath
from torch import Tensor
from typing import Tuple, List, Dict, Union


class Flareformer(nn.Module):
    def __init__(self, input_channel: int,
                 output_channel: int,
                 sfm_params: Dict[str, float],
                 mm_params: Dict[str, float],
                 window: int = 24):
        super(Flareformer, self).__init__()
        # Informer
        self.mag_encoder = nn.Sequential(*[InformerEncoderLayer(
            AttentionLayer(ProbAttention(False, factor=5, attention_dropout=mm_params["dropout"], output_attention=False),
                           d_model=mm_params["d_model"], n_heads=mm_params["h"], mix=False),
            mm_params["d_model"],
            mm_params["d_ff"],
            dropout=mm_params["dropout"],
            activation="relu"
        ) for _ in range(mm_params["N"])])

        self.phys_encoder = nn.Sequential(*[InformerEncoderLayer(
            AttentionLayer(ProbAttention(False, factor=5, attention_dropout=sfm_params["dropout"], output_attention=False),
                           d_model=sfm_params["d_model"], n_heads=sfm_params["h"], mix=False),
            sfm_params["d_model"],
            sfm_params["d_ff"],
            dropout=sfm_params["dropout"],
            activation="relu"
        ) for _ in range(sfm_params["N"])])

        # Image Feature Extractor
        self.img_embedder = ConvNeXt(in_chans=1, out_chans=mm_params["d_model"], depths=[2, 2, 2, 2], dims=[64, 128, 256, 512])

        self.linear = nn.Linear(sfm_params["d_model"] + mm_params["d_model"],
                                output_channel)

        self._linear = nn.Linear(
            window * mm_params["d_model"] * 2, sfm_params["d_model"])
        self.softmax = nn.Softmax(dim=1)

        self.img_linear = nn.Linear(
            mm_params["d_model"] * window, mm_params["d_model"])

        self.phys_linear = nn.Linear(
            sfm_params["d_model"] * window, sfm_params["d_model"])
        self.relu = torch.nn.ReLU()
        self.linear1 = torch.nn.Linear(
            input_channel, sfm_params["d_model"])  # 79 -> 128
        self.bn1 = torch.nn.BatchNorm1d(window)  # 128

    def forward(self, img_list: Tensor, feat: Tensor) -> Tuple[Tensor, Tensor]:
        # image feat
        img_feat = torch.cat([self.img_embedder(img).unsqueeze(0) for img in img_list])

        # physical feat
        phys_feat = self.linear1(feat)
        phys_feat = self.bn1(phys_feat)
        phys_feat = self.relu(phys_feat)

        # cross-attention
        Np, Nm = len(self.phys_encoder), len(self.mag_encoder)
        for i in range(max(Np, Nm)):
            merged_feat = torch.cat([phys_feat, img_feat], dim=1)
            if i < Np:
                phys_feat = self.phys_encoder[i](phys_feat, merged_feat)
            if i < Nm:
                img_feat = self.mag_encoder[i](img_feat, merged_feat)

        # Late fusion
        phys_feat = self.phys_linear(phys_feat.flatten(1))
        img_feat = self.img_linear(img_feat.flatten(1))

        x = torch.cat((phys_feat, img_feat), 1)
        output = self.linear(x)
        output = self.softmax(output)

        return output, x

    def freeze_feature_extractor(self):
        for param in self.parameters():
            param.requires_grad = False  # 重み固定

        for param in self.linear.parameters():
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

    def __init__(self, dim: int,
                 drop_path: float = 0.,
                 layer_scale_init_value: float = 1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm2(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

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

    def __init__(self,
                 in_chans: int = 3,
                 out_chans: int = 1000,
                 depths: List[int] = [3, 3, 9, 3],
                 dims: List[int] = [96, 192, 384, 768],
                 drop_path_rate: float = 0.,
                 layer_scale_init_value: float = 1e-6,
                 head_init_scale: float = 1.
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm2(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm2(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        # self.head = nn.Linear(dims[-1], num_classes)
        self.linear = nn.Linear(dims[-1], out_chans)

        self.apply(self._init_weights)
        # self.head.weight.data.mul_(head_init_scale)
        # self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m: Union[nn.Module, nn.Module, nn.Module]):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x: Tensor) -> Tensor:
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x: Tensor) -> Tensor:
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

    def __init__(self, normalized_shape: int, eps: float = 1e-6, data_format: str = "channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x: Tensor) -> Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class InformerEncoderLayer(nn.Module):
    def __init__(self,
                 attention: AttentionLayer,
                 d_model: int,
                 d_ff: int = None,
                 dropout: float = 0.1,
                 activation: str = "relu"):
        super(InformerEncoderLayer, self).__init__()

        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model,
                               out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(
            in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, q: Tensor, kv: Tensor) -> Tensor:
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

        return self.norm2(q + y)  # , attn


class PositionwiseFeedForward(torch.nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(d_model, d_ff)
        self.w_2 = torch.nn.Linear(d_ff, d_model)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class TemplateDecoder(nn.Module):
    def __init__(self,
                sfm_params: Dict[str, float],
                mm_params: Dict[str, float],
                output_flareclass: int,
                output_active: int):
        super(TemplateDecoder, self).__init__()
        
        self.linear1 = nn.Linear(sfm_params["d_model"]+mm_params["d_model"],
                                output_flareclass)
        self.softmax1 = nn.Softmax(dim=1)

        self.linear2 = nn.Linear(sfm_params["d_model"]+mm_params["d_model"],
                                output_active)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, y, feat):
        # 前半用モジュール
        """
        (1) featから過去24時間の最大太陽フレアの特徴量を取り出す
        (2) その特徴から、ルールベースで1行目を作成
        """
        class_ = self.linear1(x)
        class_ = self.softmax1(class_)

        active = self.linear2(x)
        active = self.sigmoid(active)
        
        # 前半用
        overview = self._make_overview(feat, class_, active)
        # 後半用
        predict = self._make_predict(y)
        assert len(overview) == len(predict)

        templates = []
        for idx in range(len(overview)):
            templates.append(overview[idx]+predict[idx])
        
        return class_, active, templates
    
    def _make_overview(self, feat: Tensor, class_, active):
        template = "<AA>、太陽活動は<BB>でした。"

        # TODO X線情報が含まれるindexを指定
        idx = 0
        his_X = feat[..., idx] # (8, 24)
        # 最大となるindexを得る
        history = torch.argmax(his_X, dim=1).tolist() # (8,)

        class_list = torch.argmax(class_, dim=1).tolist()
        active_list = torch.argmax(active, dim=1).tolist()
        outputs = []

        for idx in range(len(class_)):
            if active_list[idx] == 0:
                BB = "静穏"
                if class_list[idx] == 0:
                    AA = "太陽面で目立った活動は発生せず"
                elif class_list[idx] == 1:
                    AA = "活動領域でＢクラスフレアが発生しましたが"
                elif class_list[idx] == 2:
                    AA = "活動領域でＣクラスフレアが発生しましたが"
                elif class_list[idx] == 3:
                    AA = "活動領域でＭクラスフレアが発生しましたが"
                elif class_list[idx] == 4:
                    AA = "活動領域でＸクラスフレアが発生しましたが"
            
            elif active_list[idx] == 1:
                BB = "活発"
                if class_list[idx] == 0:
                    AA = "太陽面で目立った活動は発生しませんでしたが"
                elif class_list[idx] == 1:
                    AA = "活動領域でＢクラスフレアが発生し"
                elif class_list[idx] == 2:
                    AA = "活動領域でＣクラスフレアが発生し"
                elif class_list[idx] == 3:
                    AA = "活動領域でＭクラスフレアが発生し"
                elif class_list[idx] == 4:
                    AA = "活動領域でＸクラスフレアが発生し"
        
            template = template.replace("<AA>", AA).replace("<BB>", BB)
            outputs.append(template)
        return outputs
    
    def _make_predict(self, y):
        '''
            Args:
                y : FlareFormerの出力==(O, C, M, X)の4クラスの予測確率
            Return:
                テンプレ予報文の後半部分
        '''
        outputs = []
        template = "今後１日間、太陽活動は<DD>な状態が予想されます"
        labels = ["静穏", "やや活発", "活発", "非常に活発"]

        preds = torch.argmax(y, dim=1).tolist()
        for pred in preds:
            outputs.append(template.replace("<DD>", labels[pred]))
        return outputs

class ForecastFormer(nn.Module):

    def __init__(self,
                input_channel: int,
                output_channel: int,
                sfm_params: Dict[str, float],
                mm_params: Dict[str, float],
                window: int=24,
                output_flareclass: int=5,
                output_active: int=2):
        super(ForecastFormer, self).__init__()
        self.flareformer = Flareformer(
            input_channel,
            output_channel,
            sfm_params,
            mm_params,
            window,
        )
        
        # template用
        self.template_decoder = TemplateDecoder(
            sfm_params,
            mm_params,
            output_flareclass = output_flareclass,
            output_active = output_active,
        )

        # dnn用
        # self.dnn = XXX

    def forward(self, 
                img_list: Tensor, 
                feat: Tensor, 
                ) -> Tuple[Tensor, Tensor]:

        output, x = self.flareformer(img_list, feat)
        # template用
        y_class, y_active, templates = self.template_decoder(x, output, feat)
        outputs = (y_class, y_active, templates)
        
        return outputs
    
    def freeze_flareformer(self):
        for param in self.flareformer.parameters():
            param.requires_grad = False
