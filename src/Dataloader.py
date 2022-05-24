"""Dataloader for Flare Transformer"""

import torch
import torchvision.transforms as T
import numpy as np
import os
from torch.utils.data import Dataset
from skimage.transform import resize
from skimage.color import rgb2gray
from tqdm import tqdm

class CombineDataloader(Dataset):
    def __init__(self, dataloaders):
        self.dataloaders = dataloaders
        count = 0
        for dl in dataloaders: count += len(dl)
        self.all_size = count
 
    def __len__(self):
        return self.all_size

    def __getitem__(self, idx):
        for dl in self.dataloaders:
            if idx < len(dl): return dl[idx]
            else: idx -= len(dl)

        return None

class TrainDataloader(Dataset):
    def __init__(self, split, params, image_type="magnetogram", path="data/data_", augmentation=False):
        self.path = path
        self.window_size = params["window"]
        self.augmentation = augmentation

        year_split = params["year_split"]

        # get x
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
        if torch.is_tensor(idx):
            idx = idx.tolist()
        #  fancy index
        mul_x = self.img[np.asarray(self.window[idx][:self.window_size],
                                    dtype=int)]
        mul_feat = self.feat[np.asarray(self.window[idx][:self.window_size],
                                        dtype=int)]
        sample = ((mul_x - self.mean) / self.std,
                  self.label[idx],
                  mul_feat)

        return sample

    def get_multiple_year_image(self, year_dict, image_type):
        """
            concatenate data of multiple years [image]
        """
        for i, year in enumerate(range(year_dict["start"],
                                 year_dict["end"]+1)):
            data_path = self.path + str(year) + "_" + image_type + ".npy"
            print(data_path)
            image_data = np.load(data_path)
            if i == 0:
                result = image_data
            else:
                result = np.concatenate([result, image_data], axis=0)
            # print(result.shape)
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
        for i in range(ndata.shape[0] // bs + 1):
            tmp = ndata[bs*i:bs*(i+1)] - mean
            tmp = np.power(tmp, 2)
            std += np.sum(tmp)
            print("Calculating std : ", i, "/", ndata.shape[0] // bs)
        std = np.sqrt(std / len(ndata))
        return mean, std

    def set_mean(self, mean, std):
        """
            Set self.mean and self.std
        """
        self.mean = mean
        self.std = std
 

class TrainDataloader256(Dataset):
    def __init__(self, split, params, path="data/data_", augmentation=False, has_window=True, sub_bias=False, use_aia131=False, use_aia1600=False, aia_mix="channel_cat"):
        self.path = path
        self.window_size = params["window"]
        self.augmentation = augmentation
        self.has_window = has_window
        self.sub_bias = sub_bias

        year_split = params["year_split"]

        # get x
        print("loading images ...")
        img_target = []
        magnetogram = self.get_multiple_year_image(year_split[split], "magnetogram")
        img_target.append(magnetogram)
        if use_aia131:
            aia131 = self.get_multiple_year_image(year_split[split], "aia131")
            img_target.append(aia131)
        if use_aia1600:
            aia1600 = self.get_multiple_year_image(year_split[split], "aia1600")
            img_target.append(aia1600)

        if aia_mix == "channel_cat":
            self.img = torch.cat(img_target, axis=1)
        elif aia_mix == "mix":
            self.img = torch.cat(img_target, axis=0)

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
            assert False
        
        N,C,H,W = self.img.shape
        window = np.asarray(self.window[idx][:self.window_size],dtype=int)
        if N % 3 == 0:
            noise = torch.rand(C*C).cpu()  # noise in [0, 1]
            diff = torch.argsort(noise, dim=0) % C
            diff = diff[:self.window_size].detach().numpy() * (N // 3)  # magnetogram or aia131 or aia1600      
        else:
            diff = np.zeros_like(window)
        
        mul_x = self.img[window + diff]
        mul_feat = self.feat[window]

        img = torch.empty_like(mul_x)
        for c in range(C):
            img[:,c,:,:] = (mul_x[:,c,:,:] - self.mean[c]) / self.std[c]

        sample = (img,
                  self.label[idx],
                  mul_feat,
                  idx)

        # print(mul_feat.shape)
        # print(self.feat[idx[0]].shape)
        # assert False

        return sample

    def get_plain(self,idx):
        x = self.img[idx]
        feat = self.feat[idx]
        sample = ((x - self.mean) / self.std,
            self.label[idx],
            feat)

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
                    source = image_data[n,:,:,:].astype(np.uint8)
                    source = source.transpose(1,2,0)
                    # print(np.max(source))
                    # print(np.max(resize(source,(256,256))))
                    if C == 3:
                        source = rgb2gray(source)
                        source = source[:,:,np.newaxis]

                    # todo:np.uint8だとresize時に0-1で正規化されるっぽい→要検証
                    _image_data[n,:,:,:] = resize(source,(256,256)).transpose(2,0,1) 
                
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
        
        if self.sub_bias:
            result -= np.mean(result,axis=0) # todo: testもtrainと同様のバイアス画像を使うように

        # import cv2
        # x = np.empty((256,256,3))
        # result -= np.mean(result,axis=0)
        # for i in range(3): x[:,:,i] = result[0,0,:,:]
        # cv2.namedWindow('window')
        # cv2.imshow('window', x)
        # cv2.waitKey(50000)
        # cv2.destroyAllWindows()

        return torch.Tensor(result)

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
        N,C,H,W = self.img.shape
        bs = 1000000000
        mean = np.zeros(C)
        std = np.zeros(C)
        for c in range(C):
            ndata = np.ravel(self.img[:,c,:,:])
            mean[c] = np.mean(ndata)
            for i in tqdm(range(ndata.shape[0] // bs + 1)):
                tmp = ndata[bs*i:bs*(i+1)] - mean[c]
                tmp = np.power(tmp, 2)
                std[c] += np.sum(tmp)
            std[c] = np.sqrt(std[c] / len(ndata))
    
        return mean, std

    def set_mean(self, mean, std):
        """
            Set self.mean and self.std
        """
        self.mean = mean
        self.std = std


class PureImageDataset(TrainDataloader256):

    def __init__(self, split, params, path="data/data_", augmentation=False, has_window=True, sub_bias=False, use_aia131=False, use_aia1600=False, aia_mix="channel_cat"):
        super().__init__(split, params, path, augmentation, has_window, sub_bias, True, True, "mix")

        N,C,H,W = self.img.shape
        assert N % 3 == 0

        unit = N // 3
        label = np.zeros(N)
        label[unit:] += 1
        label[2*unit:] += 1
        self.label = label
        
    def __len__(self):
        """
        Returns:
            [int]: [length of sample]
        """
        return len(self.img)

    def __getitem__(self, idx):
        """
            get sample
        """
        if not self.has_window: return self.get_plain(idx)

        if torch.is_tensor(idx):
            idx = idx.tolist()

        return (self.img[idx] - self.mean) / self.std, self.label[idx]


 

# class SmoteTrainDataloader256(Dataset):
#     def __init__(self, split, params, image_type="magnetogram", path="data/data_", augmentation=False, has_window=True, sub_bias=False):
#         self.path = path
#         self.window_size = params["window"]
#         self.augmentation = augmentation
#         self.has_window = has_window
#         self.sub_bias = sub_bias

#         year_split = params["year_split"]

#         # get x
#         print("loading images ...")
#         self.img = self.get_multiple_year_image(year_split[split], image_type)
#         self.img = torch.Tensor(self.img)

#         # self.img = self.img[:, np.newaxis] # without fancy index

#         # get label
#         self.label = self.get_multiple_year_data(year_split[split], "label")

#         # get feat
#         self.feat = self.get_multiple_year_data(year_split[split],
#                                                 "feat")[:, :90]

#         # get window
#         self.window = self.get_multiple_year_window(
#             year_split[split], "window_48")[:, :self.window_size]
#         self.window = np.asarray(self.window, dtype=int)
        

#         balanced_count = len(self.label) // 4
#         class_ids = [[] for _  in range(4)]
#         for i in tqdm(range(len(self.label))):
#             class_ids[self.label[i]].append(i)

#         balanced_class_ids = [np.random.choice(class_ids[i],balanced_count,replace=False) for i in range(4)]
#         for i in range(4):
#             if 

        


#         print("img: {}".format(self.img.shape),
#               "label: {}".format(self.label.shape),
#               "feat: {}".format(self.feat.shape),
#               "window: {}".format(self.window.shape))

#     def __len__(self):
#         """
#         Returns:
#             [int]: [length of sample]
#         """
#         return len(self.label)

#     def __getitem__(self, idx):
#         """
#             get sample
#         """
#         if not self.has_window: return self.get_plain(idx)

#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#         #  fancy index
#         mul_x = self.img[np.asarray(self.window[idx][:self.window_size],
#                                     dtype=int)]
#         mul_feat = self.feat[np.asarray(self.window[idx][:self.window_size],
#                                         dtype=int)]
#         sample = ((mul_x - self.mean) / self.std,
#                   self.label[idx],
#                   mul_feat)

#         return sample

#     def get_plain(self,idx):
#         x = self.img[idx]
#         feat = self.feat[idx]
#         sample = ((x - self.mean) / self.std,
#             self.label[idx],
#             feat)

#         return sample

#     def get_multiple_year_image(self, year_dict, image_type):
#         """
#             concatenate data of multiple years [image]
#         """
#         for i, year in enumerate(tqdm(range(year_dict["start"],
#                                  year_dict["end"]+1))):
#             data_path_256 = self.path + str(year) + "_" + image_type + "_256.npy"
#             data_path_512 = self.path + str(year) + "_" + image_type + ".npy"
#             # image_data = np.load(data_path_512)
#             # print(np.max(image_data[0,0,:,:]))
#             # print(np.max(resize(image_data[0,0,:,:],(256,256))))
                
#             if not os.path.exists(data_path_256):
#                 image_data = np.load(data_path_512)
#                 N,C,H,W = image_data.shape
#                 _image_data = np.empty((N,1,256,256))
#                 for n in range(N):
#                     source = image_data[n,0,:,:].astype(np.uint8)
#                     _image_data[n,0,:,:] = resize(source,(256,256))
                
#                 image_data = _image_data
#                 np.save(data_path_256,image_data)
#             else:
#                 image_data = np.load(data_path_256)

#             for j in range(image_data.shape[0]):
#                 assert np.max(image_data[j,:,:,:]) <= 1

#             if i == 0:
#                 result = image_data
#                 # print(np.mean(result[0,0,:,:]))
#                 # import cv2
#                 # x = np.empty((256,256,3))
#                 # for i in range(3): x[:,:,i] = result[0,0,:,:]
#                 # # print(result.shape)
#                 # cv2.namedWindow('window')
#                 # cv2.imshow('window', x)
#                 # cv2.waitKey(0)
#                 # cv2.destroyAllWindows()
#             else:
#                 result = np.concatenate([result, image_data], axis=0)
        
#         if self.sub_bias:
#             result -= np.mean(result,axis=0) # todo: testもtrainと同様のバイアス画像を使うように

#         # import cv2
#         # x = np.empty((256,256,3))
#         # result -= np.mean(result,axis=0)
#         # for i in range(3): x[:,:,i] = result[0,0,:,:]
#         # cv2.namedWindow('window')
#         # cv2.imshow('window', x)
#         # cv2.waitKey(50000)
#         # cv2.destroyAllWindows()

#         return result

#     def get_multiple_year_data(self, year_dict, data_type):
#         """
#             concatenate data of multiple years [feat/label]
#         """
#         for i, year in enumerate(range(year_dict["start"],
#                                  year_dict["end"]+1)):
#             data_path = self.path + str(year) + "_" + data_type + ".csv"
#             print(data_path)
#             data = np.loadtxt(data_path, delimiter=',')
#             if i == 0:
#                 result = data
#             else:
#                 result = np.concatenate([result, data], axis=0)
#         return result

#     def get_multiple_year_window(self, year_dict, data_type):
#         """
#             concatenate data of multiple years [window]
#         """
#         num_data = 0
#         for i, year in enumerate(range(year_dict["start"],
#                                  year_dict["end"]+1)):
#             data_path = self.path + str(year) + "_" + data_type + ".csv"
#             print(data_path)
#             data = np.loadtxt(data_path, delimiter=',')
#             data += num_data
#             if i == 0:
#                 result = data
#             else:
#                 result = np.concatenate([result, data], axis=0)
#             num_data += data.shape[0]
#         return result

#     def calc_mean(self):
#         """
#             calculate mean and std of images
#         """
#         bs = 1000000000
#         ndata = np.ravel(self.img)
#         mean = np.mean(ndata)
#         std = 0
#         for i in tqdm(range(ndata.shape[0] // bs + 1)):
#             tmp = ndata[bs*i:bs*(i+1)] - mean
#             tmp = np.power(tmp, 2)
#             std += np.sum(tmp)
#             # print("Calculating std : ", i, "/", ndata.shape[0] // bs)
#         std = np.sqrt(std / len(ndata))
#         return mean, std

#     def set_mean(self, mean, std):
#         """
#             Set self.mean and self.std
#         """
#         self.mean = mean
#         self.std = std