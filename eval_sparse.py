from mae.prod.eval import *
from train_mae import FashionMnistDataLoader
import argparse
import json
# from mae.prod.datasets import *
from src.Dataloader import TrainDataloader256, TrainDatasetSparse


parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
parser.add_argument('--input_size', default=256, type=int)
parser.add_argument('--checkpoint', default=10, type=int)
parser.add_argument('--baseline', default="linear")
parser.add_argument('--grid_size', default=16, type=int)
args = parser.parse_args()

params = json.loads(open("params/params_2014.json").read())
# dl = TrainDataloader()
# img, _ = dl[0]
# img = img.transpose((1, 2, 0))
# dl = TrainDataloader256(split="train", params=params["dataset"],has_window=False)
dl = TrainDatasetSparse(split="test", params=params["dataset"],has_window=False, grid_size=args.grid_size)

mean, std = dl.calc_mean()
dl.set_mean(*dl.calc_mean())

img, _, _ = dl[-2]
print(img.shape)
img = img.transpose(0,1).transpose(1,2)

# dl = FashionMnistDataLoader()
# img, _, _ = dl[0]
# print(img.shape)
# img = img.transpose(0,1).transpose(1,2)
# print(img.shape)

# assert img.shape == (224, 224, 3)

# normalize by ImageNet mean and std
# img = img - imagenet_mean
# img = img / imagenet_std

plt.rcParams['figure.figsize'] = [5, 5]
show_image(img.clone().detach())


# This is an MAE model trained with pixels as targets for visualization
# (ViT-Large, training mask ratio=0.75)

# download checkpoint if not exist
# !wget -nc https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_large.pth
 
chkpt_dir = f'/home/katsuyuki/temp/flare_transformer/output_dir/attn/checkpoint-50-sparse32_m0.pth'
# chkpt_dir = f'/home/initial/Dropbox/flare_transformer/output_dir/attn/checkpoint-5.pth'

model_mae = prepare_model(chkpt_dir,img_size=args.input_size,baseline=args.baseline, embed_dim=64)
print('Model loaded.')

# make random mask reproducible (comment out to make it change)
torch.manual_seed(2)
print('MAE with pixel reconstruction:')
# run_one_image(img, model_mae, mean, std)
run_one_image_sp(img, model_mae)