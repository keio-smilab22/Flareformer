from mae.prod.eval import *
from train_mae import FashionMnistDataLoader
import argparse
import json
# from mae.prod.datasets import *
from src.Dataloader import TrainDataloader256, TrainDatasetSparse
from tqdm import tqdm


parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
parser.add_argument('--input_size', default=256, type=int)
parser.add_argument('--checkpoint', default=10, type=int)
parser.add_argument('--baseline', default="attn")
parser.add_argument('--grid_size', default=16, type=int)
args = parser.parse_args()

params = json.loads(open("params/params_2014.json").read())
# dl = TrainDataloader()
# img, _ = dl[0]
# img = img.transpose((1, 2, 0))
# dl = TrainDataloader256(split="train", params=params["dataset"],has_window=False)
dl = TrainDatasetSparse(split="test", params=params["dataset"],has_window=False, grid_size=args.grid_size, keep_ratio=0.1)

mean, std = dl.calc_mean()
dl.set_mean(*dl.calc_mean())


img, _, _ = dl[2]
print(img.shape)
img = img.transpose(0,1).transpose(1,2)

# normalize by ImageNet mean and std
# img = img - imagenet_mean
# img = img / imagenet_std

# plt.rcParams['figure.figsize'] = [24, 24]
# for idx, data in tqdm(enumerate(dl), total=len(dl)):
#     # if len(dl) - idx < 100:
        
#     #     img, std, _ = data
#     #     img = img.transpose(0,1).transpose(1,2)
#     #     # plt.subplot(int(len(dl)**0.5 + 1), int(len(dl)**0.5 + 1), idx+1)
#     #     plt.subplot(10, 10, len(dl)-idx+1)
#     #     show_image(img.clone().detach(), title=f"{idx}_{std:.4f}")
#     if idx < 100:
        
#         img, std, _ = data
#         img = img.transpose(0,1).transpose(1,2)
#         # plt.subplot(int(len(dl)**0.5 + 1), int(len(dl)**0.5 + 1), idx+1)
#         plt.subplot(10, 10, idx+1)
#         show_image(img.clone().detach(), title=f"{idx}_{std:.4f}")
#     else:
#         break
    
# plt.show()




# This is an MAE model trained with pixels as targets for visualization
# (ViT-Large, training mask ratio=0.75)

# download checkpoint if not exist
# !wget -nc https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_large.pth
 
chkpt_dir = f'/home/katsuyuki/temp/flare_transformer/output_dir/attn/checkpoint-50-sparse128_m0.75_0.1-128.pth'
# chkpt_dir = f'/home/initial/Dropbox/flare_transformer/output_dir/attn/checkpoint-5.pth'

model_mae = prepare_model(chkpt_dir,img_size=args.input_size,baseline=args.baseline, embed_dim=64)
print('Model loaded.')

# make random mask reproducible (comment out to make it change)
torch.manual_seed(2)
print('MAE with pixel reconstruction:')
# run_one_image(img, model_mae, mean, std)
run_one_image_sp(img, model_mae)