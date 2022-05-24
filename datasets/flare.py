"""Dataloader for Flare Transformer"""

import torch
import torchvision.transforms as T
import numpy as np
import os
from torch.utils.data import Dataset
from skimage.transform import resize
from tqdm import tqdm

class FlareDataset(Dataset):
    def __init__(self, split, params, image_type="magnetogram", path="data/data_", has_window=True):
        self.path = path
        self.window_size = params["window"]
        self.has_window = has_window

        year_split = params["year_split"]

        # get x
        print("loading images ...")
        self.img = self.get_multiple_year_image(year_split[split], image_type)
        self.img = torch.Tensor(self.img)

        # get label
        self.label = self.get_multiple_year_data(year_split[split], "label")

        # get feat
        self.feat = self.get_multiple_year_data(year_split[split],"feat")[:, :90]

        # get window
        self.window = self.get_multiple_year_window(year_split[split], "window_48")[:, :self.window_size]
        self.window = np.asarray(self.window, dtype=int)
        
        print("img: {}".format(self.img.shape),
              "label: {}".format(self.label.shape),
              "feat: {}".format(self.feat.shape),
              "window: {}".format(self.window.shape))

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

        #  fancy index
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

    def get_multiple_year_image(self, year_dict, image_type):
        """
            concatenate data of multiple years [image]
        """
        for i, year in enumerate(tqdm(range(year_dict["start"],
                                 year_dict["end"]+1))):
            data_path_256 = self.path + str(year) + "_" + image_type + "_256.npy"
            data_path_512 = self.path + str(year) + "_" + image_type + ".npy"
                
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

    def get_multiple_year_data(self, year_dict, data_type):
        """
            concatenate data of multiple years [feat/label]
        """
        result = []
        for i, year in enumerate(range(year_dict["start"],
                                 year_dict["end"]+1)):
            data_path = self.path + str(year) + "_" + data_type + ".csv"
            print(data_path)
            data = np.loadtxt(data_path, delimiter=',')
            result.append(data)
        return np.concatenate(result, axis=0)

    def get_multiple_year_window(self, year_dict, data_type):
        """
            concatenate data of multiple years [window]
        """
        num_data = 0
        result = []
        for i, year in enumerate(range(year_dict["start"],
                                 year_dict["end"]+1)):
            data_path = self.path + str(year) + "_" + data_type + ".csv"
            data = np.loadtxt(data_path, delimiter=',')
            data += num_data
            result.append(data)
            num_data += data.shape[0]
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
            # print("Calculating std : ", i, "/", ndata.shape[0] // bs)
        std = np.sqrt(std / len(ndata))
        return mean, std

    def set_mean(self, mean, std):
        """
            Set self.mean and self.std
        """
        self.mean = mean
        self.std = std

 