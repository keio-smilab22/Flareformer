import sys
sys.path.append('..')
sys.path.append('../../')

import mae.prod.models_mae
import mae.prod.models_seq_mae
import mae.prod.models_pyramid_mae
import torch
import numpy as np

import matplotlib.pyplot as plt
from mae.prod.datasets import TrainDataloader
import torch.nn.functional as F
from mae.prod.models_pyramid_mae import PyramidMaskedAutoencoderViT


def show_image(image, title=''):
    # image is [H, W, 3]
    # assert image.shape[2] == 3
    print(image.shape)
    # img = np.empty((image.shape[0],image.shape[1],3))
    # for i in range(3): img[:,:,i] = image[:,:,0]

    plt.imshow(image, cmap='gray')
    plt.title(title, fontsize=6)
    plt.axis('off') 
    return

def prepare_model(chkpt_dir, img_size=256,baseline="attn",embed_dim=512, arch='vit_for_FT', patch_size=8):
    # build model
    model = getattr(mae.prod.models_mae, arch)(img_size=img_size, baseline=baseline, embed_dim=embed_dim, patch_size=patch_size)
    # load model
    checkpoint = torch.load(chkpt_dir, map_location=torch.device('cuda'))
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    model.cuda()
    return model

def prepare_model_pyramid(chkpt_dir, img_size=256, baseline="attn",embed_dim=512, arch='vit_for_FT', patch_size=8, grid_size=32):
    # build model
    model = getattr(mae.prod.models_pyramid_mae, arch)(img_size=img_size, baseline=baseline, embed_dim=embed_dim, grid_size=32)
    # load model
    checkpoint = torch.load(chkpt_dir, map_location=torch.device('cuda'))
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    model.cuda()
    return model

def prepare_model_seq(chkpt_dir, img_size=256,baseline="attn",embed_dim=512, arch='vit_for_FT'):
    # build model
    model = getattr(mae.prod.models_seq_mae, arch)(img_size=img_size, baseline=baseline, embed_dim=embed_dim)
    # load model
    checkpoint = torch.load(chkpt_dir, map_location=torch.device('cuda'))
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    model.cuda()
    return model


def run_one_image(img, model, mean=None, std=None, mask_ratio=0.75):
    x = torch.tensor(img).cuda()
    
    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

    # run MAE
    loss, y, mask = model(x, mask_ratio=mask_ratio)
    y = model.unpatchify(y)
    mse = F.mse_loss(x, y)
    y = torch.einsum('nchw->nhwc', y).detach()
    print("loss", loss)
    print("mse", mse.item())

    # visualize the mask
    mask = mask.detach()
    # print(f"mask;{mask}")
    # mask = mask.where(mask < 0.5, torch.zeros_like(mask))
    # print(f"mask;{mask}")
    # (N, H*W, p*p*3)
    mask = mask.unsqueeze(-1).repeat(1, 1,
                                     model.patch_embed.patch_size[0]**2)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach()

    x = torch.einsum('nchw->nhwc', x)
    
    # reverse normalization
    if mean is not None and std is not None:
        print(f"x.shape: {x.shape}")
        x = x * std + mean
        y = y * std + mean

    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

    # make the plt figure larger
    plt.rcParams['figure.figsize'] = [24, 24]
    
    plt.subplot(1, 4, 1)
    show_image(x[0].cpu(), "original")

    plt.subplot(1, 4, 2)
    show_image(im_masked[0].cpu(), "masked")

    plt.subplot(1, 4, 3)
    show_image(y[0].cpu(), "reconstruction")

    plt.subplot(1, 4, 4)
    show_image(im_paste[0].cpu(), "reconstruction + visible")

    plt.show()


def run_one_image_seq(img, model, mean=None, std=None):
    x = torch.tensor(img).cuda()
    print(f"x.shape: {x.shape}")
    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nkhwc->nkchw', x)

    # run MAE
    loss, y, mask = model(x, mask_ratio=P)
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach()
    print("loss", loss)

    # visualize the mask
    mask = mask.detach()
    # (N, H*W, p*p*3)
    mask = mask.unsqueeze(-1).repeat(1, 1,
                                     model.patch_embed.patch_size[0]**2)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach()

    x = torch.einsum('nkchw->nkhwc', x)
    x1 = x[:,0,:,:,:]
    x2 = x[:,1,:,:,:]
    
    # reverse normalization
    if mean is not None and std is not None:
        print(f"x.shape: {x.shape}")
        x = x * std + mean
        y = y * std + mean

    # masked image
    im_masked = x1 * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x1 * (1 - mask) + y * mask

    # make the plt figure larger
    plt.rcParams['figure.figsize'] = [24, 24]
    
    plt.subplot(2, 4, 1)
    show_image(x1[0], "$x_{t-1}$")

    plt.subplot(2, 4, 2)
    show_image(x2[0], "$x_t$")

    plt.subplot(2, 4, 3)
    show_image(np.abs(x1[0]-x2[0]), "$ | x_t - x_{t-1}| $")
    
    plt.subplot(2, 4, 5)
    show_image(x1[0], "original")

    plt.subplot(2, 4, 6)
    show_image(im_masked[0].cpu(), "masked")

    plt.subplot(2, 4, 7)
    show_image(y[0].cpu(), "reconstruction")

    plt.subplot(2, 4, 8)
    show_image(im_paste[0].cpu(), "reconstruction + visible")

    plt.show()


def run_one_image_sp(img, model, mean=None, std=None, mask_ratio=0.75):
    x = torch.tensor(img).cuda()
    
    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

    # run MAE
    loss, y, mask = model(x, mask_ratio=mask_ratio)
    y = model.unpatchify(y)
    mse = F.mse_loss(x, y)
    y = torch.einsum('nchw->nhwc', y).detach()
    print("loss", loss)
    print("mse", mse.item())

    # visualize the mask
    mask = mask.detach()
    # (N, H*W, p*p*3)
    mask = mask.unsqueeze(-1).repeat(1, 1,
                                     model.patch_embed.patch_size[0]**2)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach()

    x = torch.einsum('nchw->nhwc', x)
    
    # reverse normalization
    if mean is not None and std is not None:
        print(f"x.shape: {x.shape}")
        x = x * std + mean
        y = y * std + mean

    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

    # make the plt figure larger
    plt.rcParams['figure.figsize'] = [24, 24]
    
    plt.subplot(1, 3, 1)
    show_image(x[0].cpu(), "original")

    plt.subplot(1, 3, 2)
    show_image(im_masked[0].cpu(), "masked")

    plt.subplot(1, 3, 3)
    show_image(y[0].cpu(), "reconstruction")

    # plt.subplot(1, 3, 3)
    # show_image(im_paste[0].cpu(), "reconstruction + visible")

    plt.show()


class MaskedAutoEncoder:
    def __init__(self,baseline,embed_dim):
        chkpt_dir = f'/home/katsuyuki/temp/flare_transformer/output_dir/{baseline}/checkpoint-50-64d4b_base-16.pth' # パス注意
        self.model = prepare_model(chkpt_dir,baseline=baseline,embed_dim=embed_dim)
        self.dim = self.model.embed_dim

    def get_model(self):
        return self.model

    def train(self,flag=True):
        self.model.train(flag)

    def encode(self,img):
        # if img.shape[1] == 1:
        #     img = torch.cat((img,img,img),1)
        
        # assert img.shape[1] == 3 and img.shape[-1] == img.shape[-2], "shape: {}".format(img.shape)
        x = img
        latent, _, _ = self.model.forward_encoder_std(x, 0.5)

        return latent[:,0,:] # CLSトークンのみ使用


class PyramidMaskedAutoEncoder:
    def __init__(self,baseline,embed_dim):
        chkpt_dir = f'/home/katsuyuki/temp/flare_transformer/output_dir/attn/checkpoint-50-pyramid-32.pth' # パス注意
        self.model:PyramidMaskedAutoencoderViT = prepare_model_pyramid(chkpt_dir,baseline=baseline,embed_dim=embed_dim)
        self.dim = self.model.mae2.embed_dim

    def get_model(self):
        return self.model

    def train(self,flag=True):
        self.model.train(flag)

    def encode(self,img):

        rows = img.shape[2]//self.model.grid_size
        cols = img.shape[3]//self.model.grid_size

        imgs_list, std_list = self.model.grid_dividing_image(img, rows=rows, cols=cols)

        # imgs_list, ids_restore_std = self.model.std_masking(imgs_list, std_list, keep_ratio=0.75)

        latent_list, mask_list, ids_restore_list = self.model.forward_encoder(imgs_list, mask_ratio=0.5) # latent_list: (Gridの数, N, L, D)
        pred_list = self.model.forward_decoder(latent_list=latent_list, mask_list=mask_list, ids_restore_list=ids_restore_list)

        # average over grids
        # latent = torch.mean(torch.stack(latent_list, dim=0), dim=0)

        # if use second stage
        img_merged = self.model.unpatchify(pred_list)
        img_merged = self.model.reshape_image(img_merged, rows=rows, cols=cols)
        img_merged = img_merged.to(img.device)


        latent, mask, ids_restore = self.model.mae2.forward_encoder(img_merged, mask_ratio=0.5)
        # print("latent.shape", latent.shape)



        return latent[:,0,:] # CLSトークンのみ使用



# dl = TrainDataloader()
# img, _ = dl[0]
# mae_model = MaskedAutoEncoder()
# mae_model.encode(img)


# exit(0)

# dl = TrainDataloader()
# img, _ = dl[0]
# img = img.transpose((1, 2, 0))
# # img = resize(img,(224,224))
# print(img.shape)

# # assert img.shape == (224, 224, 3)

# # normalize by ImageNet mean and std
# # img = img - imagenet_mean
# # img = img / imagenet_std

# plt.rcParams['figure.figsize'] = [5, 5]
# show_image(torch.tensor(img))


# # This is an MAE model trained with pixels as targets for visualization
# # (ViT-Large, training mask ratio=0.75)

# # download checkpoint if not exist
# # !wget -nc https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_large.pth

# chkpt_dir = '/home/initial/Dropbox/flare_transformer/mae/prod/output_dir/checkpoint-15.pth'
# model_mae = prepare_model(chkpt_dir, 'mae_vit_large_patch16')
# print('Model loaded.')

# # make random mask reproducible (comment out to make it change)
# torch.manual_seed(2)
# print('MAE with pixel reconstruction:')
# run_one_image(img, model_mae)
