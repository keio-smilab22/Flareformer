# python train_mae.py --input_size=256 --epoch=20 --batch_size=47 --baseline=linear --wandb | tee linear.log
# python train_mae.py --input_size=256 --epoch=20 --batch_size=27 --baseline=lambda --wandb | tee lambda.log
# python train_mae.py --input_size=256 --epoch=20 --batch_size=13 --baseline=attn --wandb | tee attn.log

# python train_sparse.py --input_size=${i} --epoch=20 --batch_size= --baseline=attn --wandb 
for i in 0.7 0.8 0.9 1
do
    python train_sparse.py --baseline attn --wandb --batch_size 8192 --dim 16 --input_size 16 --name sparse16_m0 --epoch 50 --mask_ratio 0 --grid_size 16 --keep_ratio ${i}
done

for i in 0.1 0.2 0.5 1
do
    python train_sparse.py --baseline attn --wandb --batch_size 2048 --dim 64 --input_size 32 --name sparse32_m0 --epoch 50 --mask_ratio 0 --grid_size 32 --keep_ratio ${i}
done

for i in 0.1 0.2 0.5 1
do
    python train_sparse.py --baseline attn --wandb --batch_size 32 --dim 64 --input_size 128 --name sparse128_m0 --epoch 50 --mask_ratio 0 --grid_size 128 --keep_ratio ${i}
done