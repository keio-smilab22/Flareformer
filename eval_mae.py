from mae.prod.eval import *

dl = TrainDataloader()
img, _ = dl[0]
img = img.transpose((1, 2, 0))
# img = resize(img,(224,224))
print(img.shape)

# assert img.shape == (224, 224, 3)

# normalize by ImageNet mean and std
# img = img - imagenet_mean
# img = img / imagenet_std

plt.rcParams['figure.figsize'] = [5, 5]
show_image(torch.tensor(img))


# This is an MAE model trained with pixels as targets for visualization
# (ViT-Large, training mask ratio=0.75)

# download checkpoint if not exist
# !wget -nc https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_large.pth
 
chkpt_dir = '/home/initial/Dropbox/flare_transformer/output_dir/checkpoint-1.pth'
model_mae = prepare_model(chkpt_dir)
print('Model loaded.')

# make random mask reproducible (comment out to make it change)
torch.manual_seed(2)
print('MAE with pixel reconstruction:')
run_one_image(img, model_mae)
