import sys
sys.path.append('..')
sys.path.append('../../')

import mae.prod.models_mae
import torch
import numpy as np

import matplotlib.pyplot as plt
from mae.prod.datasets import TrainDataloader

P = 0.75

def show_image(image, title=''):
    # image is [H, W, 3]
    # assert image.shape[2] == 3
    print(image.shape)
    img = np.empty((image.shape[0],image.shape[1],3))
    for i in range(3): img[:,:,i] = image[:,:,0]

    plt.imshow(img)
    plt.title(title, fontsize=16)
    plt.axis('off') 
    return

def prepare_model(chkpt_dir, img_size=256,baseline="attn",embed_dim=512, arch='vit_for_FT'):
    # build model
    model = getattr(mae.prod.models_mae, arch)(img_size=img_size, baseline=baseline, embed_dim=embed_dim)
    # load model
    checkpoint = torch.load(chkpt_dir, map_location=torch.device('cuda'))
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    model.cuda()
    return model


def run_one_image(img, model):
    x = torch.tensor(img).cuda()
    
    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

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

    x = torch.einsum('nchw->nhwc', x)

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


class MaskedAutoEncoder:
    def __init__(self,baseline,embed_dim):
        chkpt_dir = f'/home/initial/Dropbox/flare_transformer/output_dir/{baseline}/checkpoint-30.pth' # パス注意
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
        latent, _, _ = self.model.forward_encoder(x, 0)

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
