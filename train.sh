#!/bin/sh

# seq-mae: patch=16
# train: 
#   1stage: lr = 2.1e-6 , 
#   2stage: lr = 8e-6 , epoch=10

for year in 2014 2015 2016 2017
do
    # MEM: 256 → 12G (2017)

    # 20stageのlr調整したので注意
    python train.py --params params/params_${year}.json --dim=128 --baseline=attn --enc_depth=4 --dec_depth=4 --baseline=attn --warmup_epochs=5 --imbalance --lr_stage2=0.0000021 --wandb
done

# mask_ratio=0.20 にしたので注意！！！！！！！！！！！！！
# MEM: 32　→　9G (2017)
# python train_mae.py --params params/params_2014.json --input_size=256 --epoch=100 --dim=128 --batch_size=32 --enc_depth=4 --dec_depth=4 --baseline=attn --target=seq --mask_ratio=0.20 --output_dir=output_dir --wandb