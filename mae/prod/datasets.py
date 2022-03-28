from torch.utils.data import Dataset
import numpy as np


class TrainDataloader(Dataset):
    def __init__(
            self,
            image_type="magnetogram",
            path="data/data_",
            augmentation=False):
        self.path = path

        # get x
        # self.img = self.get_multiple_year_image( image_type)
        # self.img = torch.Tensor(self.img)

        # print("img: {}".format(self.img.shape))

    def __len__(self):
        """
        Returns:
            [int]: [length of sample]
        """
        # return 61315 # all
        return 29247 # 2010-2013
        # return 100  # debug

    def __getitem__(self, idx):
        """
            get sample
        """
        img = np.load(f"data/{idx}.npy")
        assert np.max(img) <= 1
        # resized = torch.Tensor(resize(img[0,:,:],(256,256))).unsqueeze(0)
        # img = torch.cat((resized,resized,resized),0) # ViT処理のためにチャネル方向に増やしておく
        # print(img.shape)

        return img, 0
