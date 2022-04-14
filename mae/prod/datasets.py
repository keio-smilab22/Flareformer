from torch.utils.data import Dataset
import numpy as np
import torch
import torchvision.transforms as T
import numpy as np
import os
from torch.utils.data import Dataset
from skimage.transform import resize
from tqdm import tqdm


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


class TrainDataloader256(Dataset):
    def __init__(self, split, params, image_type="magnetogram", path="data/data_", augmentation=False, has_window=True):
        self.path = path
        self.window_size = params["window"]
        self.augmentation = augmentation
        self.has_window = has_window

        year_split = params["year_split"]

        # get x
        print("loading images ...")
        self.img = self.get_multiple_year_image(year_split[split], image_type)
        self.img = torch.Tensor(self.img)
        if self.augmentation:
            transform = T.Compose([T.ToPILImage(),
                                    T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                                    T.PILToTensor()])

            S,C,H,W = self.img.shape
            # img = torch.empty((S*2,C,H,W))
            for i in range(S):
                self.img[i,:,:,:] = transform(self.img[i,0,:,:])

        # self.img = self.img[:, np.newaxis] # without fancy index

        # get label
        self.label = self.get_multiple_year_data(year_split[split], "label")

        # get feat
        self.feat = self.get_multiple_year_data(year_split[split],
                                                "feat")[:, :90]

        # get window
        self.window = self.get_multiple_year_window(
            year_split[split], "window_48")[:, :self.window_size]
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

        if torch.is_tensor(idx):
            idx = idx.tolist()
        #  fancy index
        mul_x = self.img[np.asarray(self.window[idx][:self.window_size],
                                    dtype=int)]
        mul_feat = self.feat[np.asarray(self.window[idx][:self.window_size],
                                        dtype=int)]
        sample = ((mul_x - self.mean) / self.std,
                  0)

        return sample

    def get_plain(self,idx):
        x = self.img[idx]
        feat = self.feat[idx]
        sample = ((x - self.mean) / self.std,
            0)

        return sample

    def get_multiple_year_image(self, year_dict, image_type):
        """
            concatenate data of multiple years [image]
        """
        for i, year in enumerate(tqdm(range(year_dict["start"],
                                 year_dict["end"]+1))):
            data_path_256 = self.path + str(year) + "_" + image_type + "_256.npy"
            data_path_512 = self.path + str(year) + "_" + image_type + ".npy"
            # image_data = np.load(data_path_512)
            # print(np.max(image_data[0,0,:,:]))
            # print(np.max(resize(image_data[0,0,:,:],(256,256))))
                
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
                # print(np.mean(result[0,0,:,:]))
                # import cv2
                # x = np.empty((256,256,3))
                # for i in range(3): x[:,:,i] = result[0,0,:,:]
                # # print(result.shape)
                # cv2.namedWindow('window')
                # cv2.imshow('window', x)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
            else:
                result = np.concatenate([result, image_data], axis=0)
            
        return result

    def get_multiple_year_data(self, year_dict, data_type):
        """
            concatenate data of multiple years [feat/label]
        """
        for i, year in enumerate(range(year_dict["start"],
                                 year_dict["end"]+1)):
            data_path = self.path + str(year) + "_" + data_type + ".csv"
            print(data_path)
            data = np.loadtxt(data_path, delimiter=',')
            if i == 0:
                result = data
            else:
                result = np.concatenate([result, data], axis=0)
        return result

    def get_multiple_year_window(self, year_dict, data_type):
        """
            concatenate data of multiple years [window]
        """
        num_data = 0
        for i, year in enumerate(range(year_dict["start"],
                                 year_dict["end"]+1)):
            data_path = self.path + str(year) + "_" + data_type + ".csv"
            print(data_path)
            data = np.loadtxt(data_path, delimiter=',')
            data += num_data
            if i == 0:
                result = data
            else:
                result = np.concatenate([result, data], axis=0)
            num_data += data.shape[0]
        return result

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
