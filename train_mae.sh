# python train_mae.py --input_size=256 --epoch=20 --batch_size=47 --baseline=linear --wandb | tee linear.log
# python train_mae.py --input_size=256 --epoch=20 --batch_size=27 --baseline=lambda --wandb | tee lambda.log
# python train_mae.py --input_size=256 --epoch=20 --batch_size=13 --baseline=attn --wandb | tee attn.log

# python train_sparse.py --input_size=${i} --epoch=20 --batch_size= --baseline=attn --wandb 
# for i in 0.1 0.2 0.5 1
# do
#     python train_sparse.py --baseline attn --wandb --batch_size 8 --dim 64 --input_size 256 --name sparse256_m0_${i} --epoch 50 --mask_ratio 0 --grid_size 256 --keep_ratio ${i}
# done

# for i in 0.1 0.2 0.5 1
# do
#     python train_sparse.py --baseline attn --wandb --batch_size 32 --dim 64 --input_size 128 --name sparse128_m0_${i} --epoch 50 --mask_ratio 0 --grid_size 128 --keep_ratio ${i}
# done

# for i in 0.1 0.2 0.5 1
# do
#     python train_sparse.py --baseline attn --wandb --batch_size 2048 --dim 64 --input_size 64 --name sparse64_m0_${i} --epoch 50 --mask_ratio 0 --grid_size 64 --keep_ratio ${i}
# done

# for i in 0.75 0.5 0.25 0.1 
# do
#     for p in 2 4 8
#     do
#         python train_sparse.py --baseline attn --wandb --batch_size 256 --dim 64 --input_size 32 --name sparse32_m${i}_p${p} --epoch 50 --mask_ratio ${i} --grid_size 32 --keep_ratio 0.1 --patch_size ${p}
#     done
# done

# for i in 0.75 0.5 0.25 0.1 
# do
#     for p in 2
#     do
#         python train_sparse.py --baseline attn --wandb --batch_size 32 --dim 64 --input_size 64 --name sparse64_m${i}_p${p} --epoch 50 --mask_ratio ${i} --grid_size 64 --keep_ratio 0.1 --patch_size ${p}
#     done
# done

# for i in 0.25 0.5
# do
#     python train_mae.py --baseline attn --batch_size 8 --dim 64 --epoch 20 --mask_ratio ${i} --wandb --name new_pyra_${i} --grid_size 64
# done

# python train_mae.py --baseline attn --batch_size 8 --dim 64 --epoch 50 --mask_ratio 0.5 --wandb --name 64d4b_stdwise --patch_size 8 --do_pyramid

# python train_mae.py --baseline attn --batch_size 8 --dim 64 --epoch 50 --mask_ratio 0.5 --wandb --name 64d4b_base --patch_size 8


for r in 0.1 0.2 0.25 0.5
do
    python train_mae.py --baseline attn --batch_size 64 --dim 64 --epoch 50 --mask_ratio ${r} --wandb --name pyramid_1st${r}_2nd0.75 --grid_size 32
done