"""Dataloader for Flare Transformer"""

import torch
import torchvision.transforms as T
import numpy as np
import os
from torch.utils.data import Dataset
from skimage.transform import resize
from tqdm import tqdm

class FlareDataset(Dataset):
    def __init__(self, dataset_type, params, image_type="magnetogram", path="data/data_", has_window=True):
        self.path = path
        self.window_size = params["window"]
        self.has_window = has_window

        year_split = params["year_split"][dataset_type]
        start_year, end_year = year_split["start"], year_split["end"]

        # get x
        print("Loading images ...")
        self.img = self.get_multiple_year_image(start_year, end_year, image_type)
        self.img = torch.Tensor(self.img)

        # get label
        print("\nLoading labels ...")
        self.label = self.get_multiple_year_data(start_year, end_year, "label")

        # get feat
        print("\nLoading features ...")
        self.feat = self.get_multiple_year_data(start_year, end_year,"feat")[:, :90]

        # get window
        print("\nLoading windows ...")
        self.window = self.get_multiple_year_window(start_year, end_year, "window_48")[:, :self.window_size]
        self.window = np.asarray(self.window, dtype=int)
        
        print(f"img: {self.img.shape}\n",
              f"label: {self.label.shape}\n",
              f"feat: {self.feat.shape}\n",
              f"window: {self.window.shape}\n")

    def __len__(self):
        """
        Returns:
            [int]: [length of sample]
        """
        return len(self.label)

    def __getitem__(self, idx):
        """
            get sample
        """
        if not self.has_window: return self.get_plain(idx)

        window = np.asarray(self.window[idx][:self.window_size],dtype=int)
        imgs, feats = self.img[window], self.feat[window]
        imgs = (imgs - self.mean) / self.std
        x = (imgs, feats)
        y = self.label[idx]

        return x, y, idx

    def get_plain(self,idx):
        imgs, feats = self.img[idx], self.feat[idx]
        imgs = (imgs - self.mean) / self.std
        x = (imgs,feats)
        y = self.label[idx]

        return x, y, idx

    def get_multiple_year_image(self, start_year, end_year, image_type):
        """
            concatenate data of multiple years [image]
        """
        for i, year in enumerate(tqdm(range(start_year,
                                 end_year+1))):
            data_path_256 = f"{self.path}{year}_{image_type}_256.npy"
            data_path_512 = f"{self.path}{year}_{image_type}.npy"
                
            if not os.path.exists(data_path_256):
                image_data = np.load(data_path_512)
                N,C,H,W = image_data.shape
                _image_data = np.empty((N,1,256,256))
                for n in range(N):
                    source = image_data[n,0,:,:].astype(np.uint8)
                    _image_data[n,0,:,:] = resize(source,(256,256))
                
                image_data = _image_data
                np.save(data_path_256,image_data)
            else:
                image_data = np.load(data_path_256)

            for j in range(image_data.shape[0]):
                assert np.max(image_data[j,:,:,:]) <= 1

            if i == 0:
                result = image_data
            else:
                result = np.concatenate([result, image_data], axis=0)


        return result

    def get_multiple_year_data(self, start_year, end_year, data_type):
        """
            concatenate data of multiple years [feat/label]
        """
        result = []
        for year in tqdm(range(start_year, end_year+1)):
            data_path = f"{self.path}{year}_{data_type}.csv"
            data = np.loadtxt(data_path, delimiter=',')
            result.append(data)
        return np.concatenate(result, axis=0)

    def get_multiple_year_window(self, start_year, end_year, data_type):
        """
            concatenate data of multiple years [window]
        """
        result, N = [], 0
        for year in tqdm(range(start_year, end_year+1)):
            data_path = f"{self.path}{year}_{data_type}.csv"
            data_path = self.path + str(year) + "_" + data_type + ".csv"
            data = np.loadtxt(data_path, delimiter=',')
            result.append(data + N)
            N += data.shape[0]
        return np.concatenate(result, axis=0)

    def calc_mean(self):
        """
            calculate mean and std of images
        """
        bs = 1000000000
        ndata = np.ravel(self.img)
        mean = np.mean(ndata)
        std = 0
        for i in tqdm(range(ndata.shape[0] // bs + 1)):
            tmp = ndata[bs*i:bs*(i+1)] - mean
            tmp = np.power(tmp, 2)
            std += np.sum(tmp)
        std = np.sqrt(std / len(ndata))
        return mean, std

    def set_mean(self, mean, std):
        """
            Set self.mean and self.std
        """
        self.mean = mean
        self.std = std

 