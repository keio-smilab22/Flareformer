import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from models.attn import ProbAttention, AttentionLayer
from timm.models.layers import trunc_normal_, DropPath
from torch import Tensor
from typing import Tuple, List, Dict, Union
from torch.nn.modules.container import Sequential
from torch.nn.modules.conv import Conv2d

from utils.ema_module import EMAModule, EMAModuleConfig
from random import sample

class FlareFormer(nn.Module):
    def __init__(self, input_channel: int,
                 output_channel: int,
                 sfm_params: Dict[str, float],
                 mm_params: Dict[str, float],
                 window: int = 24):
        super(FlareFormer, self).__init__()

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

class FlareformerEncoder(nn.Module):
    def __init__(self, input_channel: int,
                 output_channel: int,
                 sfm_params: Dict[str, float],
                 mm_params: Dict[str, float],
                 window: int = 24):
        super(FlareformerEncoder, self).__init__()

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
        # print("encoder:", x.shape)
        return x


class FlareFormerWithPCL(nn.Module):
    def __init__(self, input_channel: int,
                 output_channel: int,
                 sfm_params: Dict[str, float],
                 mm_params: Dict[str, float],
                 window: int = 24,
                 train_type: str = "pretrain"):
        super(FlareFormerWithPCL, self).__init__()

        self.encoder = FlareformerEncoder(input_channel, output_channel, sfm_params, mm_params, window)
        self.linear = nn.Linear(sfm_params["d_model"] + mm_params["d_model"],
                                output_channel)
        self.softmax = nn.Softmax(dim=1)

        self.train_type = train_type
        self.change_train_type(train_type)

    def change_train_type(self,train_type):
        self.train_type = train_type
        for params in self.linear.parameters():
            params.requires_grad = train_type != "pretrain"

    def forward(self, img_list: Tensor, feat: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.encoder(img_list, feat)

        if self.train_type == "pretrain":
            return x
        else:
            output = self.linear(x)
            output = self.softmax(output)
            return output, x

    def freeze_feature_extractor(self):
        for param in self.encoder.parameters():
            param.requires_grad = False  # 重み固定

        for param in self.linear.parameters():
            param.requires_grad = True

    def step_ema(self):
        self.ema.step(self.encoder)




class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=256, r=16384, m=0.999, T=0.1, mlp=False):
        """
        dim: feature dimension (default: 128)
        r: queue size; number of negative samples/prototypes (default: 16384)
        m: momentum for updating key encoder (default: 0.999)
        T: softmax temperature 
        mlp: whether to use mlp projection
        """
        super(MoCo, self).__init__()

        self.r = r
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = copy.deepcopy(base_encoder)
        self.encoder_k = copy.deepcopy(base_encoder)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, r))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.r % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.r  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        # x_gather = concat_all_gather(x)
        x_gather = x
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        # torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        idx_this = idx_shuffle

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, feat_q, im_k=None, feat_k=None, is_eval=False, cluster_result=None, index=None):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
            is_eval: return momentum embeddings (used for clustering)
            cluster_result: cluster assignments, centroids, and density
            index: indices for training samples
        Output:
            logits, targets, proto_logits, proto_targets
        """
        
        if is_eval:
            k = self.encoder_k(im_q,feat_q)  
            k = nn.functional.normalize(k, dim=1)            
            return k
        
        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            # im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k,feat_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # # undo shuffle
            # k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute query features
        q = self.encoder_q(im_q,feat_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)
        
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: Nxr
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+r)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)
        
        # prototypical contrast
        if cluster_result is not None:  
            proto_labels = []
            proto_logits = []
            for n, (im2cluster,prototypes,density) in enumerate(zip(cluster_result['im2cluster'],cluster_result['centroids'],cluster_result['density'])):
                # get positive prototypes
                # print(im2cluster.shape,index.max())
                pos_proto_id = im2cluster[index]
                pos_prototypes = prototypes[pos_proto_id]    
                
                # sample negative prototypes
                all_proto_id = [i for i in range(im2cluster.max()+1)]       
                neg_proto_id = set(all_proto_id)-set(pos_proto_id.tolist())
                if len(neg_proto_id) > self.r:
                    neg_proto_id = sample(neg_proto_id,self.r) #sample r negative prototypes
                else:
                    neg_proto_id = list(neg_proto_id) 
                neg_prototypes = prototypes[neg_proto_id]

                proto_selected = torch.cat([pos_prototypes,neg_prototypes],dim=0)
                
                # compute prototypical logits
                logits_proto = torch.mm(q,proto_selected.t())
                
                # targets for prototype assignment
                labels_proto = torch.linspace(0, q.size(0)-1, steps=q.size(0)).long().cuda()
                
                # scaling temperatures for the selected prototypes
                temp_proto = density[torch.cat([pos_proto_id,torch.LongTensor(neg_proto_id).cuda()],dim=0)]  
                logits_proto /= temp_proto
                
                proto_labels.append(labels_proto)
                proto_logits.append(logits_proto)
            return logits, labels, proto_logits, proto_labels
        else:
            return logits, labels, None, None


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output




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
