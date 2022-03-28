import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from skimage.transform import resize


class Preprocess(Dataset):
    def __init__(
            self,
            image_type="magnetogram",
            path="data/data_",
            augmentation=False):
        self.path = path

        # get x
        self.img = self.get_multiple_year_image(
            image_type, start=2010, end=2013)
        self.img = torch.Tensor(self.img)

        print("img: {}".format(self.img.shape))

    def __len__(self):
        """
        Returns:
            [int]: [length of sample]
        """
        return self.img.shape[0]
        # return 100

    def run(self):
        for idx in tqdm(range(self.img.shape[0])):
            # print(np.max(self.img[0,0,:,:].cpu().numpy()))
            # print(self.img[0,0,:,:].cpu().numpy().dtype)
            # print(np.max(resize(self.img[0,0,:,:].cpu().numpy(),(256,256))))
            img = self.img[idx, 0, :, :].cpu().numpy().astype(np.uint8)
            img = torch.Tensor(resize(img, (256, 256))).unsqueeze(0)
            assert np.max(img.cpu().numpy()) <= 1. , f"{np.max(img.cpu().numpy())}"
            # img = torch.cat((img, img, img), 0)  # ViT処理のためにチャネル方向に増やしておく
            np.save(f"data/{idx}.npy", img)

    def get_multiple_year_image(self, image_type, start, end):
        """
            concatenate data of multiple years [image]
        """
        for i, year in enumerate(range(start, end + 1)):
            data_path = "./" + self.path + \
                str(year) + "_" + image_type + ".npy"
            image_data = np.load(data_path)
            if i == 0:
                result = image_data
            else:
                result = np.concatenate([result, image_data], axis=0)
        return result


if __name__ == "__main__":
    Preprocess().run()
