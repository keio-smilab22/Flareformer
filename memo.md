

- lambda → 20,038,848

```
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
MaskedAutoencoderViT                     --
├─PatchEmbed: 1-1                        --
│    └─Conv2d: 2-1                       8,320
│    └─Identity: 2-2                     --
├─ModuleList: 1-2                        --
│    └─Block: 2-3                        --
│    │    └─LayerNorm: 3-1               256
│    │    └─LambdaLayer: 3-2             22,240
│    │    └─Identity: 3-3                --
│    │    └─Identity: 3-4                --
│    │    └─LayerNorm: 3-5               256
│    │    └─Mlp: 3-6                     131,712
│    │    └─Identity: 3-7                --
│    │    └─Identity: 3-8                --
│    └─Block: 2-4                        --
│    │    └─LayerNorm: 3-9               256
│    │    └─LambdaLayer: 3-10            22,240
│    │    └─Identity: 3-11               --
│    │    └─Identity: 3-12               --
│    │    └─LayerNorm: 3-13              256
│    │    └─Mlp: 3-14                    131,712
│    │    └─Identity: 3-15               --
│    │    └─Identity: 3-16               --
│    └─Block: 2-5                        --
│    │    └─LayerNorm: 3-17              256
│    │    └─LambdaLayer: 3-18            22,240
│    │    └─Identity: 3-19               --
│    │    └─Identity: 3-20               --
│    │    └─LayerNorm: 3-21              256
│    │    └─Mlp: 3-22                    131,712
│    │    └─Identity: 3-23               --
│    │    └─Identity: 3-24               --
│    └─Block: 2-6                        --
│    │    └─LayerNorm: 3-25              256
│    │    └─LambdaLayer: 3-26            22,240
│    │    └─Identity: 3-27               --
│    │    └─Identity: 3-28               --
│    │    └─LayerNorm: 3-29              256
│    │    └─Mlp: 3-30                    131,712
│    │    └─Identity: 3-31               --
│    │    └─Identity: 3-32               --
│    └─Block: 2-7                        --
│    │    └─LayerNorm: 3-33              256
│    │    └─LambdaLayer: 3-34            22,240
│    │    └─Identity: 3-35               --
│    │    └─Identity: 3-36               --
│    │    └─LayerNorm: 3-37              256
│    │    └─Mlp: 3-38                    131,712
│    │    └─Identity: 3-39               --
│    │    └─Identity: 3-40               --
│    └─Block: 2-8                        --
│    │    └─LayerNorm: 3-41              256
│    │    └─LambdaLayer: 3-42            22,240
│    │    └─Identity: 3-43               --
│    │    └─Identity: 3-44               --
│    │    └─LayerNorm: 3-45              256
│    │    └─Mlp: 3-46                    131,712
│    │    └─Identity: 3-47               --
│    │    └─Identity: 3-48               --
│    └─Block: 2-9                        --
│    │    └─LayerNorm: 3-49              256
│    │    └─LambdaLayer: 3-50            22,240
│    │    └─Identity: 3-51               --
│    │    └─Identity: 3-52               --
│    │    └─LayerNorm: 3-53              256
│    │    └─Mlp: 3-54                    131,712
│    │    └─Identity: 3-55               --
│    │    └─Identity: 3-56               --
│    └─Block: 2-10                       --
│    │    └─LayerNorm: 3-57              256
│    │    └─LambdaLayer: 3-58            22,240
│    │    └─Identity: 3-59               --
│    │    └─Identity: 3-60               --
│    │    └─LayerNorm: 3-61              256
│    │    └─Mlp: 3-62                    131,712
│    │    └─Identity: 3-63               --
│    │    └─Identity: 3-64               --
│    └─Block: 2-11                       --
│    │    └─LayerNorm: 3-65              256
│    │    └─LambdaLayer: 3-66            22,240
│    │    └─Identity: 3-67               --
│    │    └─Identity: 3-68               --
│    │    └─LayerNorm: 3-69              256
│    │    └─Mlp: 3-70                    131,712
│    │    └─Identity: 3-71               --
│    │    └─Identity: 3-72               --
│    └─Block: 2-12                       --
│    │    └─LayerNorm: 3-73              256
│    │    └─LambdaLayer: 3-74            22,240
│    │    └─Identity: 3-75               --
│    │    └─Identity: 3-76               --
│    │    └─LayerNorm: 3-77              256
│    │    └─Mlp: 3-78                    131,712
│    │    └─Identity: 3-79               --
│    │    └─Identity: 3-80               --
│    └─Block: 2-13                       --
│    │    └─LayerNorm: 3-81              256
│    │    └─LambdaLayer: 3-82            22,240
│    │    └─Identity: 3-83               --
│    │    └─Identity: 3-84               --
│    │    └─LayerNorm: 3-85              256
│    │    └─Mlp: 3-86                    131,712
│    │    └─Identity: 3-87               --
│    │    └─Identity: 3-88               --
│    └─Block: 2-14                       --
│    │    └─LayerNorm: 3-89              256
│    │    └─LambdaLayer: 3-90            22,240
│    │    └─Identity: 3-91               --
│    │    └─Identity: 3-92               --
│    │    └─LayerNorm: 3-93              256
│    │    └─Mlp: 3-94                    131,712
│    │    └─Identity: 3-95               --
│    │    └─Identity: 3-96               --
├─LayerNorm: 1-3                         256
├─Linear: 1-4                            66,048
├─ModuleList: 1-5                        --
│    └─Block: 2-15                       --
│    │    └─LayerNorm: 3-97              1,024
│    │    └─LambdaLayer: 3-98            157,840
│    │    └─Identity: 3-99               --
│    │    └─Identity: 3-100              --
│    │    └─LayerNorm: 3-101             1,024
│    │    └─Mlp: 3-102                   2,099,712
│    │    └─Identity: 3-103              --
│    │    └─Identity: 3-104              --
│    └─Block: 2-16                       --
│    │    └─LayerNorm: 3-105             1,024
│    │    └─LambdaLayer: 3-106           157,840
│    │    └─Identity: 3-107              --
│    │    └─Identity: 3-108              --
│    │    └─LayerNorm: 3-109             1,024
│    │    └─Mlp: 3-110                   2,099,712
│    │    └─Identity: 3-111              --
│    │    └─Identity: 3-112              --
│    └─Block: 2-17                       --
│    │    └─LayerNorm: 3-113             1,024
│    │    └─LambdaLayer: 3-114           157,840
│    │    └─Identity: 3-115              --
│    │    └─Identity: 3-116              --
│    │    └─LayerNorm: 3-117             1,024
│    │    └─Mlp: 3-118                   2,099,712
│    │    └─Identity: 3-119              --
│    │    └─Identity: 3-120              --
│    └─Block: 2-18                       --
│    │    └─LayerNorm: 3-121             1,024
│    │    └─LambdaLayer: 3-122           157,840
│    │    └─Identity: 3-123              --
│    │    └─Identity: 3-124              --
│    │    └─LayerNorm: 3-125             1,024
│    │    └─Mlp: 3-126                   2,099,712
│    │    └─Identity: 3-127              --
│    │    └─Identity: 3-128              --
│    └─Block: 2-19                       --
│    │    └─LayerNorm: 3-129             1,024
│    │    └─LambdaLayer: 3-130           157,840
│    │    └─Identity: 3-131              --
│    │    └─Identity: 3-132              --
│    │    └─LayerNorm: 3-133             1,024
│    │    └─Mlp: 3-134                   2,099,712
│    │    └─Identity: 3-135              --
│    │    └─Identity: 3-136              --
│    └─Block: 2-20                       --
│    │    └─LayerNorm: 3-137             1,024
│    │    └─LambdaLayer: 3-138           157,840
│    │    └─Identity: 3-139              --
│    │    └─Identity: 3-140              --
│    │    └─LayerNorm: 3-141             1,024
│    │    └─Mlp: 3-142                   2,099,712
│    │    └─Identity: 3-143              --
│    │    └─Identity: 3-144              --
│    └─Block: 2-21                       --
│    │    └─LayerNorm: 3-145             1,024
│    │    └─LambdaLayer: 3-146           157,840
│    │    └─Identity: 3-147              --
│    │    └─Identity: 3-148              --
│    │    └─LayerNorm: 3-149             1,024
│    │    └─Mlp: 3-150                   2,099,712
│    │    └─Identity: 3-151              --
│    │    └─Identity: 3-152              --
│    └─Block: 2-22                       --
│    │    └─LayerNorm: 3-153             1,024
│    │    └─LambdaLayer: 3-154           157,840
│    │    └─Identity: 3-155              --
│    │    └─Identity: 3-156              --
│    │    └─LayerNorm: 3-157             1,024
│    │    └─Mlp: 3-158                   2,099,712
│    │    └─Identity: 3-159              --
│    │    └─Identity: 3-160              --
├─LayerNorm: 1-6                         1,024
├─Linear: 1-7                            32,832
=================================================================
Total params: 20,038,848
Trainable params: 20,038,848
Non-trainable params: 0
=================================================================
```


- vit → 20,038,848

```
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
MaskedAutoencoderViT                     --
├─PatchEmbed: 1-1                        --
│    └─Conv2d: 2-1                       8,320
│    └─Identity: 2-2                     --
├─ModuleList: 1-2                        --
│    └─Block: 2-3                        --
│    │    └─LayerNorm: 3-1               256
│    │    └─Attention: 3-2               66,048
│    │    └─Identity: 3-3                --
│    │    └─Identity: 3-4                --
│    │    └─LayerNorm: 3-5               256
│    │    └─Mlp: 3-6                     131,712
│    │    └─Identity: 3-7                --
│    │    └─Identity: 3-8                --
│    └─Block: 2-4                        --
│    │    └─LayerNorm: 3-9               256
│    │    └─Attention: 3-10              66,048
│    │    └─Identity: 3-11               --
│    │    └─Identity: 3-12               --
│    │    └─LayerNorm: 3-13              256
│    │    └─Mlp: 3-14                    131,712
│    │    └─Identity: 3-15               --
│    │    └─Identity: 3-16               --
│    └─Block: 2-5                        --
│    │    └─LayerNorm: 3-17              256
│    │    └─Attention: 3-18              66,048
│    │    └─Identity: 3-19               --
│    │    └─Identity: 3-20               --
│    │    └─LayerNorm: 3-21              256
│    │    └─Mlp: 3-22                    131,712
│    │    └─Identity: 3-23               --
│    │    └─Identity: 3-24               --
│    └─Block: 2-6                        --
│    │    └─LayerNorm: 3-25              256
│    │    └─Attention: 3-26              66,048
│    │    └─Identity: 3-27               --
│    │    └─Identity: 3-28               --
│    │    └─LayerNorm: 3-29              256
│    │    └─Mlp: 3-30                    131,712
│    │    └─Identity: 3-31               --
│    │    └─Identity: 3-32               --
│    └─Block: 2-7                        --
│    │    └─LayerNorm: 3-33              256
│    │    └─Attention: 3-34              66,048
│    │    └─Identity: 3-35               --
│    │    └─Identity: 3-36               --
│    │    └─LayerNorm: 3-37              256
│    │    └─Mlp: 3-38                    131,712
│    │    └─Identity: 3-39               --
│    │    └─Identity: 3-40               --
│    └─Block: 2-8                        --
│    │    └─LayerNorm: 3-41              256
│    │    └─Attention: 3-42              66,048
│    │    └─Identity: 3-43               --
│    │    └─Identity: 3-44               --
│    │    └─LayerNorm: 3-45              256
│    │    └─Mlp: 3-46                    131,712
│    │    └─Identity: 3-47               --
│    │    └─Identity: 3-48               --
│    └─Block: 2-9                        --
│    │    └─LayerNorm: 3-49              256
│    │    └─Attention: 3-50              66,048
│    │    └─Identity: 3-51               --
│    │    └─Identity: 3-52               --
│    │    └─LayerNorm: 3-53              256
│    │    └─Mlp: 3-54                    131,712
│    │    └─Identity: 3-55               --
│    │    └─Identity: 3-56               --
│    └─Block: 2-10                       --
│    │    └─LayerNorm: 3-57              256
│    │    └─Attention: 3-58              66,048
│    │    └─Identity: 3-59               --
│    │    └─Identity: 3-60               --
│    │    └─LayerNorm: 3-61              256
│    │    └─Mlp: 3-62                    131,712
│    │    └─Identity: 3-63               --
│    │    └─Identity: 3-64               --
│    └─Block: 2-11                       --
│    │    └─LayerNorm: 3-65              256
│    │    └─Attention: 3-66              66,048
│    │    └─Identity: 3-67               --
│    │    └─Identity: 3-68               --
│    │    └─LayerNorm: 3-69              256
│    │    └─Mlp: 3-70                    131,712
│    │    └─Identity: 3-71               --
│    │    └─Identity: 3-72               --
│    └─Block: 2-12                       --
│    │    └─LayerNorm: 3-73              256
│    │    └─Attention: 3-74              66,048
│    │    └─Identity: 3-75               --
│    │    └─Identity: 3-76               --
│    │    └─LayerNorm: 3-77              256
│    │    └─Mlp: 3-78                    131,712
│    │    └─Identity: 3-79               --
│    │    └─Identity: 3-80               --
│    └─Block: 2-13                       --
│    │    └─LayerNorm: 3-81              256
│    │    └─Attention: 3-82              66,048
│    │    └─Identity: 3-83               --
│    │    └─Identity: 3-84               --
│    │    └─LayerNorm: 3-85              256
│    │    └─Mlp: 3-86                    131,712
│    │    └─Identity: 3-87               --
│    │    └─Identity: 3-88               --
│    └─Block: 2-14                       --
│    │    └─LayerNorm: 3-89              256
│    │    └─Attention: 3-90              66,048
│    │    └─Identity: 3-91               --
│    │    └─Identity: 3-92               --
│    │    └─LayerNorm: 3-93              256
│    │    └─Mlp: 3-94                    131,712
│    │    └─Identity: 3-95               --
│    │    └─Identity: 3-96               --
├─LayerNorm: 1-3                         256
├─Linear: 1-4                            66,048
├─ModuleList: 1-5                        --
│    └─Block: 2-15                       --
│    │    └─LayerNorm: 3-97              1,024
│    │    └─Attention: 3-98              1,050,624
│    │    └─Identity: 3-99               --
│    │    └─Identity: 3-100              --
│    │    └─LayerNorm: 3-101             1,024
│    │    └─Mlp: 3-102                   2,099,712
│    │    └─Identity: 3-103              --
│    │    └─Identity: 3-104              --
│    └─Block: 2-16                       --
│    │    └─LayerNorm: 3-105             1,024
│    │    └─Attention: 3-106             1,050,624
│    │    └─Identity: 3-107              --
│    │    └─Identity: 3-108              --
│    │    └─LayerNorm: 3-109             1,024
│    │    └─Mlp: 3-110                   2,099,712
│    │    └─Identity: 3-111              --
│    │    └─Identity: 3-112              --
│    └─Block: 2-17                       --
│    │    └─LayerNorm: 3-113             1,024
│    │    └─Attention: 3-114             1,050,624
│    │    └─Identity: 3-115              --
│    │    └─Identity: 3-116              --
│    │    └─LayerNorm: 3-117             1,024
│    │    └─Mlp: 3-118                   2,099,712
│    │    └─Identity: 3-119              --
│    │    └─Identity: 3-120              --
│    └─Block: 2-18                       --
│    │    └─LayerNorm: 3-121             1,024
│    │    └─Attention: 3-122             1,050,624
│    │    └─Identity: 3-123              --
│    │    └─Identity: 3-124              --
│    │    └─LayerNorm: 3-125             1,024
│    │    └─Mlp: 3-126                   2,099,712
│    │    └─Identity: 3-127              --
│    │    └─Identity: 3-128              --
│    └─Block: 2-19                       --
│    │    └─LayerNorm: 3-129             1,024
│    │    └─Attention: 3-130             1,050,624
│    │    └─Identity: 3-131              --
│    │    └─Identity: 3-132              --
│    │    └─LayerNorm: 3-133             1,024
│    │    └─Mlp: 3-134                   2,099,712
│    │    └─Identity: 3-135              --
│    │    └─Identity: 3-136              --
│    └─Block: 2-20                       --
│    │    └─LayerNorm: 3-137             1,024
│    │    └─Attention: 3-138             1,050,624
│    │    └─Identity: 3-139              --
│    │    └─Identity: 3-140              --
│    │    └─LayerNorm: 3-141             1,024
│    │    └─Mlp: 3-142                   2,099,712
│    │    └─Identity: 3-143              --
│    │    └─Identity: 3-144              --
│    └─Block: 2-21                       --
│    │    └─LayerNorm: 3-145             1,024
│    │    └─Attention: 3-146             1,050,624
│    │    └─Identity: 3-147              --
│    │    └─Identity: 3-148              --
│    │    └─LayerNorm: 3-149             1,024
│    │    └─Mlp: 3-150                   2,099,712
│    │    └─Identity: 3-151              --
│    │    └─Identity: 3-152              --
│    └─Block: 2-22                       --
│    │    └─LayerNorm: 3-153             1,024
│    │    └─Attention: 3-154             1,050,624
│    │    └─Identity: 3-155              --
│    │    └─Identity: 3-156              --
│    │    └─LayerNorm: 3-157             1,024
│    │    └─Mlp: 3-158                   2,099,712
│    │    └─Identity: 3-159              --
│    │    └─Identity: 3-160              --
├─LayerNorm: 1-6                         1,024
├─Linear: 1-7                            32,832
=================================================================
Total params: 27,706,816
Trainable params: 27,706,816
Non-trainable params: 0
=================================================================
```

# patch=8, batch=10

patch_size=8, embed_dim=128, depth=12, num_heads=8, # embed_dim % num_heads == 0 にしないとだめなので注意
decoder_embed_dim=128, decoder_depth=8, decoder_num_heads=8,
mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), baseline="lambda", **kwargs)


- ViT → 8分 / MEM: 10.5GB
- Lambda → 5分 / MEM: 4.5GB


# patch=4, batch=1

- ViT → MEM: 13.6GB
- Lambda → MEM: 3.8 GB


# patch=4, batch=10

- Lambda → MEM: 8GB

# patch=16, batch=128

- ViT → 4,048,384
- Lambda → 3,172,224

- lossが1以下に下がらない
    - 共通部分はバグってない？
        - vit → OK
    - encoderで特徴量をつかめていないのか / decoder
    - decoderの次元を上げてみる = encoderで信頼性のある特徴を抽出できているなら精度出るはず
        - loss減った
        - 画像がスパースすぎて意味ないかも
            - → 他のデータセットでやってみる → fashion mnistなど



====== baseline : linear ======
batch_size=512 => NG
batch_size=256 => NG
batch_size=128 => NG
batch_size=64 => OK
batch_size=96 => NG
batch_size=80 => OK
batch_size=88 => NG
batch_size=84 => OK
batch_size=86 => OK
batch_size=87 => OK

=> batch_size = 87

====== baseline : attn ======
batch_size=512 => NG
batch_size=256 => NG
batch_size=128 => NG
batch_size=64 => NG
batch_size=32 => NG
batch_size=16 => OK
batch_size=24 => NG
batch_size=20 => OK
batch_size=22 => OK
batch_size=23 => OK

=> batch_size = 23

====== baseline : lambda ======
batch_size=512 => NG
batch_size=256 => NG
batch_size=128 => NG
batch_size=64 => NG
batch_size=32 => OK
batch_size=48 => OK
batch_size=56 => NG
batch_size=52 => NG
batch_size=50 => NG
batch_size=49 => OK

=> batch_size = 49

