# python train_mae.py --input_size=256 --epoch=20 --batch_size=47 --baseline=linear --wandb | tee linear.log
python train_mae.py --input_size=256 --epoch=20 --batch_size=27 --baseline=lambda --wandb | tee lambda.log
# python train_mae.py --input_size=256 --epoch=20 --batch_size=13 --baseline=attn --wandb | tee attn.log