
# 最後 --wandb 忘れない！！
for year in `seq 2014 2017`
do
    # without cRT
    python train.py --model_name id24_${year}_ablation --params params/params_${year}.json --warmup_epochs=5 --epoch_for_2stage=10 --wandb
    
    # N = 2
    python train.py --model_name id32_N2_${year}_ablation --params params/paramsN2_${year}.json --warmup_epochs=5 --epoch_for_2stage=10 --imbalance --wandb
    
    # N = 3
    python train.py --model_name id32_N3_${year}_ablation --params params/paramsN3_${year}.json --warmup_epochs=5 --epoch_for_2stage=10 --imbalance --wandb

    # ResNet
    python train.py --model FlareFormerWithCNN --model_name id17a_${year}_ablation --params params/params_${year}.json --warmup_epochs=5 --epoch_for_2stage=10 --imbalance --wandb
    
    # Vanilla Transformer
    python train.py --model FlareFormerWithVanillaTransformer --model_name resnet_${year}_ablation --params params/params_${year}.json --warmup_epochs=5 --epoch_for_2stage=10 --imbalance --wandb
done