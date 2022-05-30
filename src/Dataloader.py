"""Dataloader for Flare Transformer"""

from statistics import mean
from typing import Dict
import torch
import torchvision.transforms as T
import numpy as np
import os
from torch.utils.data import Dataset
from skimage.transform import resize
import cv2
from tqdm import tqdm
import pandas as pd
from utils.tools import StandardScaler
from utils.timefeatures import time_features

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
                  self.label[idx],
                  mul_feat)
        # sample = (mul_x,
        #           self.label[idx],
        #           mul_feat)

        return sample

    def get_plain(self,idx):
        x = self.img[idx]
        feat = self.feat[idx]
        # sample = ((x - self.mean) / self.std,
        #     self.label[idx],
        #     feat)
        sample = (x,
            self.label[idx],
            feat)


        return sample

    def get_multiple_year_image(self, year_dict, image_type):
        """
            concatenate data of multiple years [image]
        """
        for i, year in enumerate(tqdm(range(year_dict["start"],
                                 year_dict["end"]+1))):
            data_path_256 = self.path + str(year) + "_" + image_type + "_256_gomi.npy"
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
                    print(f"source mean: {np.mean(source)}")
                    # _image_data[n,0,:,:] = resize(source,(256,256))
                    _image_data[n,0,:,:] = cv2.resize(source,(256,256))
                    print(f"resize mean: {np.mean(_image_data[n,0,:,:])}")
                image_data = _image_data
                np.save(data_path_256,image_data)
                print("save image data to {}".format(data_path_256))
            else:
                image_data = np.load(data_path_256)

            for j in range(image_data.shape[0]):
                assert np.max(image_data[j,:,:,:]) <= 255

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



class TrainDatasetSparse(Dataset):
    def __init__(self, split, params, image_type="magnetogram", path="data/data_", augmentation=False, has_window=True, grid_size=16, keep_ratio=0.1):
        self.path = path
        self.window_size = params["window"]
        self.augmentation = augmentation
        self.has_window = has_window

        year_split = params["year_split"]

        # get x
        print("loading images ...")
        self.img, self.std = self.get_multiple_year_image_dumm(year_split[split], image_type, grid_size, keep_ratio)
        self.img = torch.Tensor(self.img)
        self.img_std = torch.Tensor(self.std)
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
        # self.label = self.get_multiple_year_data(year_split[split], "label")

        # get feat
        # self.feat = self.get_multiple_year_data(year_split[split],
                                                # "feat")[:, :90]

        # get window
        # self.window = self.get_multiple_year_window(
        #     year_split[split], "window_48")[:, :self.window_size]
        # self.window = np.asarray(self.window, dtype=int)

        # print("img: {}".format(self.img.shape),
        #       "label: {}".format(self.label.shape),
        #       "feat: {}".format(self.feat.shape),
        #       "window: {}".format(self.window.shape))
        print("img: {}".format(self.img.shape))
           

    def __len__(self):
        """
        Returns:
            [int]: [length of sample]
        """
        # return len(self.label)
        return self.img.shape[0]


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
        # mul_feat = self.feat[np.asarray(self.window[idx][:self.window_size],
                                        # dtype=int)]
        # sample = ((mul_x - self.mean) / self.std,
                #   self.label[idx],
                #   mul_feat)
        # sample = (mul_x,
        #           self.label[idx],
        #           mul_feat)
        sample = (mul_x, 0, 0)

        return sample

    def get_plain(self,idx):
        x = self.img[idx]
        std = self.img_std[idx]
        # feat = self.feat[idx]
        # sample = ((x - self.mean) / self.std,
        #     self.label[idx],
        #     feat)
        # sample = (x,
        #     self.label[idx],
        #     feat)
        sample = (x, std, 0)
        return sample

    def get_multiple_year_image(self, year_dict, image_type):
        """
            concatenate data of multiple years [image]
        """
        for i, year in enumerate(tqdm(range(year_dict["start"],
                                 year_dict["end"]+1))):
            data_path_sp = self.path + str(year) + "_" + image_type + "_sparse.npy"
            data_path_512 = self.path + str(year) + "_" + image_type + ".npy"
            # image_data = np.load(data_path_512)
            # print(np.max(image_data[0,0,:,:]))
            # print(np.max(resize(image_data[0,0,:,:],(256,256))))
                
            if not os.path.exists(data_path_sp):
                image_data = np.load(data_path_512)
                N,C,H,W = image_data.shape
                cropped = image_data[:,:,H//4:3*H//4,W//4:3*W//4]
                NC,CC,HC,WC = cropped.shape
                
                size = (16,16)
                rows = int(HC/size[0])
                cols = int(WC/size[1])
                
                chunks = []
                selected_chunks = []

                for row_img in np.array_split(cropped, rows, axis=2):
                    for col_img in np.array_split(row_img, cols, axis=3):
                        # calculate mean and std
                        mean = np.mean(col_img)
                        std = np.std(col_img)
                        chunks.append(col_img)
                        
                chunks = np.array(chunks)
                print(chunks.shape) # (256, 2, 1, 16, 16)
                chunks = chunks.transpose(1,0,2,3,4)
                chunks = chunks.reshape(chunks.shape[0]*chunks.shape[1],chunks.shape[2], chunks.shape[3],chunks.shape[4])
                np.save(data_path_sp, chunks)
                # for n in range(N):
                #     source = cropped[n,0,:,:].astype(np.uint8)
                #     print(f"source mean: {np.mean(source)}")
                #     _image_data[n,0,:,:] = cv2.resize(source,(256,256))
                #     print(f"resize mean: {np.mean(_image_data[n,0,:,:])}")
                # image_data = _image_data
                # np.save(data_path_sp,image_data)
                print("save image data to {}".format(data_path_sp))
            else:
                image_data = np.load(data_path_sp)

            for j in range(image_data.shape[0]):
                assert np.max(image_data[j,:,:,:]) <= 255

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

    def get_multiple_year_image_dumm(self, year_dict, image_type, grid_size:int=16, keep_ratio:float=0.1):
        """
            concatenate data of multiple years [image]
        """
        for i, year in enumerate(tqdm(range(year_dict["start"],
                                 year_dict["end"]+1))):
            # data_path_sp = self.path + str(year) + "_" + image_type + "_sparse_" + str(grid_size) +  ".npy"
            data_path_sp = f"{self.path}{year}_{image_type}_sparse_{grid_size}_{keep_ratio}.npy"
            data_path_std = f"{self.path}{year}_{image_type}_sparse_{grid_size}_{keep_ratio}_std.npy"

            data_path_512 = self.path + str(year) + "_" + image_type + ".npy"
            # image_data = np.load(data_path_512)
            # print(np.max(image_data[0,0,:,:]))
            # print(np.max(resize(image_data[0,0,:,:],(256,256))))
                
            if not os.path.exists(data_path_sp) or not os.path.exists(data_path_std):
                image_data = np.load(data_path_512)
                N,C,H,W = image_data.shape
                cropped = image_data[:,:,H//4:3*H//4,W//4:3*W//4]
                _,_,HC,WC = cropped.shape
                
                size = (grid_size,grid_size)
                rows = int(HC/size[0])
                cols = int(WC/size[1])
                
                chunks = []
                stds = []
                len_keep = int(N*rows*cols*keep_ratio) if rows*cols*keep_ratio > 0  else N
                for n in range(N):
                    for row_img in np.array_split(cropped[n,:,:,:], rows, axis=1):
                        for col_img in np.array_split(row_img, cols, axis=2):
                            # calculate mean and std
                            mean = np.mean(col_img)
                            std = np.std(col_img)
                            # print(f"mean: {mean}, std: {std}")
                            chunks.append(col_img)
                            stds.append(std)
                chunks = np.array(chunks)
                stds = np.array(stds)
                ids = np.argsort(stds, axis=0)
                print(f"ids: {ids}")
                ids = ids[::-1]
                # keep the first subset
                ids_keep = ids[:len_keep]
                print(f"before chunks shape: {chunks.shape}") # (256*N, 1, 16, 16)
                chunks = chunks[ids_keep]
                stds = stds[ids_keep]
                print(f"after chunks shape: {chunks.shape}") # (256*N, 1, 16, 16)
                # chunks = chunks.transpose(1,0,2,3,4)
                # chunks = chunks.reshape(chunks.shape[0]*chunks.shape[1],chunks.shape[2], chunks.shape[3],chunks.shape[4])
                np.save(data_path_sp, chunks)
                np.save(data_path_std, stds)
                # for n in range(N):
                #     source = cropped[n,0,:,:].astype(np.uint8)
                #     print(f"source mean: {np.mean(source)}")
                #     _image_data[n,0,:,:] = cv2.resize(source,(256,256))
                #     print(f"resize mean: {np.mean(_image_data[n,0,:,:])}")
                image_data = chunks
                std_data = stds
                # np.save(data_path_sp,image_data)
                print("save image data to {}".format(data_path_sp))
            else:
                image_data = np.load(data_path_sp)
                std_data = np.load(data_path_std)

            for j in range(image_data.shape[0]):
                assert np.max(image_data[j,:,:,:]) <= 255

            if i == 0:
                result = image_data
                result_std = std_data
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
                result_std = np.concatenate([result_std, std_data], axis=0)
            
        return result, result_std


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


class TrainDatasetForPyramid(Dataset):
    def __init__(self, split, params, image_type="magnetogram", path="data/data_", augmentation=False, has_window=True, grid_size=128):
        self.path = path
        self.window_size = params["window"]
        self.augmentation = augmentation
        self.has_window = has_window

        year_split = params["year_split"]

        # get x
        print("loading images ...")
        self.img = self.get_multiple_year_image(year_split[split], image_type, grid_size)
        self.img = torch.Tensor(self.img)
        if self.augmentation:
            transform = T.Compose([T.ToPILImage(),
                                    T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                                    T.PILToTensor()])

            S,C,H,W = self.img.shape
            # img = torch.empty((S*2,C,H,W))
            for i in range(S):
                self.img[i,:,:,:] = transform(self.img[i,0,:,:])

        
        print("img: {}".format(self.img.shape))
           

    def __len__(self):
        """
        Returns:
            [int]: [length of sample]
        """
        # return len(self.label)
        return self.img.shape[0]


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
        # mul_feat = self.feat[np.asarray(self.window[idx][:self.window_size],
                                        # dtype=int)]
        # sample = ((mul_x - self.mean) / self.std,
                #   self.label[idx],
                #   mul_feat)
        # sample = (mul_x,
        #           self.label[idx],
        #           mul_feat)
        sample = (mul_x, 0, 0)

        return sample

    def get_plain(self,idx):
        x = self.img[idx]
        # feat = self.feat[idx]
        # sample = ((x - self.mean) / self.std,
        #     self.label[idx],
        #     feat)
        # sample = (x,
        #     self.label[idx],
        #     feat)
        sample = (x, 0, 0)
        return sample

    def get_multiple_year_image(self, year_dict, image_type, grid_size:int=16):
        """
            concatenate data of multiple years [image]
        """
        for i, year in enumerate(tqdm(range(year_dict["start"],
                                 year_dict["end"]+1))):
            # data_path_sp = self.path + str(year) + "_" + image_type + "_sparse_" + str(grid_size) +  ".npy"
            data_path_sp = f"{self.path}{year}_{image_type}_pyramid_{grid_size}.npy"
            data_path_512 = self.path + str(year) + "_" + image_type + ".npy"
            print(data_path_sp)
            # image_data = np.load(data_path_512)
            # print(np.max(image_data[0,0,:,:]))
            # print(np.max(resize(image_data[0,0,:,:],(256,256))))
                
            if not os.path.exists(data_path_sp):
                image_data = np.load(data_path_512)
                N,C,H,W = image_data.shape
                # cropped = image_data[:,:,H//4:3*H//4,W//4:3*W//4]
                # _,_,HC,WC = cropped.shape
                
                size = (grid_size,grid_size)
                rows = int(H/size[0])
                cols = int(W/size[1])
                
                chunks = []
                stds = []
                # len_keep = int(N*rows*cols*keep_ratio) if rows*cols*keep_ratio > 0  else N
                for n in range(N):
                    for row_img in np.array_split(image_data[n,:,:,:], rows, axis=1):
                        for col_img in np.array_split(row_img, cols, axis=2):
                            # calculate mean and std
                            mean = np.mean(col_img)
                            std = np.std(col_img)
                            chunks.append(col_img)
                            stds.append(std)
                chunks = np.array(chunks)
                stds = np.array(stds)
                # ids = np.argsort(stds, axis=0)
                # ids = ids[::-1]
                # keep the first subset
                # ids_keep = ids[:len_keep]
                print(f"before chunks shape: {chunks.shape}") # (256*N, 1, 16, 16)
                # chunks = chunks[ids_keep]
                print(f"after chunks shape: {chunks.shape}") # (256*N, 1, 16, 16)
                # chunks = chunks.transpose(1,0,2,3,4)
                # chunks = chunks.reshape(chunks.shape[0]*chunks.shape[1],chunks.shape[2], chunks.shape[3],chunks.shape[4])
                np.save(data_path_sp, chunks)
                # for n in range(N):
                #     source = cropped[n,0,:,:].astype(np.uint8)
                #     print(f"source mean: {np.mean(source)}")
                #     _image_data[n,0,:,:] = cv2.resize(source,(256,256))
                #     print(f"resize mean: {np.mean(_image_data[n,0,:,:])}")
                image_data = chunks
                # np.save(data_path_sp,image_data)
                print("save image data to {}".format(data_path_sp))
            else:
                image_data = np.load(data_path_sp)

            for j in range(image_data.shape[0]):
                assert np.max(image_data[j,:,:,:]) <= 255

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




class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', feat_path='data_all_v2.csv', magnetogram_path='data_magnetogram_256.npy',
                 target='logXmax1h', scale=True, inverse=False, timeenc=0, freq='h', cols=None, year=2014):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        self.root_path = root_path
        self.feat_path = feat_path
        self.magnetogram_path = magnetogram_path
        self.year = year

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.feat_path))
        # TODO check if name is correct
        data_magnetogram = np.load(os.path.join(self.root_path,self.magnetogram_path))

        '''
        df_raw.columns: ['Time', ...(other features), target feature]
        '''
        # cols = list(df_raw.columns); 
        # print(df_raw.columns)

        if self.cols:
            cols=self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('Time')
        df_raw = df_raw[['Time']+cols+[self.target]]

        year_dict = {
            2015:{'num_train':31439, 'num_test':8760, 'end_index':48960},
        }
        # num_train = int(len(df_raw)*0.7)
        # num_test = int(len(df_raw)*0.2)
        num_train = 31439 # 2013-12-31 23:00
        # num_train = 22679
        num_test = 8760
        num_test_2 = len(df_raw) - num_train

        end_index = 48961 # 2016-01-01 00:00
        
        # num_vali = len(df_raw) - num_train - num_test
        num_vali = end_index - num_train - num_test
        # num_vali = 31440 - 22680
        
        print(f'train\n{df_raw[0:num_train]}')
        print(f'val\n{df_raw[num_train-self.seq_len:num_train+num_vali]}')
        print(f'test\n{df_raw[len(df_raw)-num_test-self.seq_len:len(df_raw)]}')

        # border1s = [0, num_train-self.seq_len, len(df_raw)-num_test-self.seq_len] #[train, val, test]
        # border2s = [num_train, num_train+num_vali, len(df_raw)] #[train, val, test]
        border1s = [0, num_train-self.seq_len, end_index-num_test-self.seq_len] #[train, val, test]
        border2s = [num_train, num_train+num_vali, end_index] #[train, val, test]
        
        # border1s = [0, len(df_raw)-num_test-self.seq_len, num_train-self.seq_len] #[train, val, test]
        # border2s = [num_train, len(df_raw), num_train+num_vali]
        # border1s = [0, num_train-self.seq_len, len(df_raw)-num_test_2-self.seq_len]
        # border2s = [num_train, num_train+num_vali, len(df_raw)]
        # border1s = [0, num_train-self.seq_len, 0]
        # border2s = [num_train, num_train+num_vali, num_test]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        # TODO testをtrainを入れる

        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        # print(f"df_raw: {df_raw}")
        # print(f"df_data {df_data}")

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]] # NOTE Only train dataset to be normalized
            self.scaler.fit(train_data.values) # trainのmean, stdを計算
            data = self.scaler.transform(df_data.values) # 全データをtrainのmean, stdで標準化
            # print(f"data: {data.shape}")
            
        else:
            data = df_data.values
            # print(f"data: {data}")
            
        df_stamp = df_raw[['Time']][border1:border2]
        df_stamp['Time'] = pd.to_datetime(df_stamp["Time"], format='%Y-%m-%d %H:%M:%S')
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        
        self.data_x = data[border1:border2]
        self.data_magnetogram = data_magnetogram[border1:border2]

        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        # print(self.data_x)
        # print(self.data_y)
    
    def __getitem__(self, index):
        s_begin = index # start index
        s_end = s_begin + self.seq_len # end index
        r_begin = s_end - self.label_len # decoder start index, label_lenだけ前のデータを使う
        r_end = r_begin + self.label_len + self.pred_len
        # Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

        # print(f"s_begin: {s_begin}")
        # print(f"s_end: {s_end}")
        # print(f"r_begin: {r_begin}")
        # print(f"r_end: {r_end}")

        seq_x = self.data_x[s_begin:s_end]
        # print(f"self.data_x.shape: {self.data_x.shape}")
        # print(f"self.data_magnetogram.shape: {self.data_magnetogram.shape}")

        seq_magnetogram = self.data_magnetogram[s_begin:s_end]
        # print(f"index: {index}, seq_x: {seq_x.shape}, seq_magnetogram: {seq_magnetogram.shape}")

        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end] #2:28
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        # seq_y_mark = seq_y_mark[-1]

        
        return seq_x, seq_magnetogram, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom_Stddev(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', feat_path='data_all_v2.csv', magnetogram_path='data_magnetogram_256.npy',
                 target='logXmax1h', scale=True, inverse=False, timeenc=0, freq='h', cols=None, year=2014):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        self.root_path = root_path
        self.feat_path = feat_path
        self.magnetogram_path = magnetogram_path
        self.year = year

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.feat_path))
        # TODO check if name is correct
        data_magnetogram = np.load(os.path.join(self.root_path,self.magnetogram_path))

        '''
        df_raw.columns: ['Time', ...(other features), target feature]
        '''
        # cols = list(df_raw.columns); 
        # print(df_raw.columns)

        if self.cols:
            cols=self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('Time')
        df_raw = df_raw[['Time']+cols+[self.target]]

        year_dict = {
            2015:{'num_train':31440, 'num_test':8760, 'end_index':48960},
            2016:{'num_train':40200, 'num_test':8784, 'end_index':57744},
            2017:{'num_train':48960, 'num_test':8692, 'end_index':66437},
        }
        # num_train = int(len(df_raw)*0.7)
        # num_test = int(len(df_raw)*0.2)
        num_train = year_dict[self.year]['num_train']
        # num_train = 22679
        num_test = year_dict[self.year]['num_test']
        num_test_2 = len(df_raw) - num_train

        end_index = year_dict[self.year]['end_index']
        # num_vali = len(df_raw) - num_train - num_test
        num_vali = end_index - num_train - num_test
        # num_vali = 31440 - 22680
        
        print(f'train\n{df_raw[0:num_train]}')
        print(f'val\n{df_raw[num_train-self.seq_len:num_train+num_vali]}')
        print(f'test\n{df_raw[end_index-num_test-self.seq_len:end_index]}')

        # border1s = [0, num_train-self.seq_len, len(df_raw)-num_test-self.seq_len] #[train, val, test]
        # border2s = [num_train, num_train+num_vali, len(df_raw)] #[train, val, test]
        border1s = [0, num_train-self.seq_len, end_index-num_test-self.seq_len] #[train, val, test]
        border2s = [num_train, num_train+num_vali, end_index] #[train, val, test]
        
        # border1s = [0, len(df_raw)-num_test-self.seq_len, num_train-self.seq_len] #[train, val, test]
        # border2s = [num_train, len(df_raw), num_train+num_vali]
        # border1s = [0, num_train-self.seq_len, len(df_raw)-num_test_2-self.seq_len]
        # border2s = [num_train, num_train+num_vali, len(df_raw)]
        # border1s = [0, num_train-self.seq_len, 0]
        # border2s = [num_train, num_train+num_vali, num_test]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        # TODO testをtrainを入れる

        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        # print(f"df_raw: {df_raw}")
        # print(f"df_data {df_data}")

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]] # NOTE Only train dataset to be normalized
            self.scaler.fit(train_data.values) # trainのmean, stdを計算
            data = self.scaler.transform(df_data.values) # 全データをtrainのmean, stdで標準化
            # print(f"data: {data.shape}")
            
        else:
            data = df_data.values
            # print(f"data: {data}")
            
        df_stamp = df_raw[['Time']][border1:border2]
        df_stamp['Time'] = pd.to_datetime(df_stamp["Time"], format='%Y-%m-%d %H:%M:%S')
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        
        self.data_x = data[border1:border2]
        self.data_magnetogram = data_magnetogram[border1:border2]

        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        # print(self.data_x)
        # print(self.data_y)
    
    def __getitem__(self, index):
        s_begin = index # start index
        s_end = s_begin + self.seq_len # end index
        r_begin = s_end - self.label_len # decoder start index, label_lenだけ前のデータを使う
        r_end = r_begin + self.label_len + self.pred_len
        # Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

        # print(f"s_begin: {s_begin}")
        # print(f"s_end: {s_end}")
        # print(f"r_begin: {r_begin}")
        # print(f"r_end: {r_end}")

        seq_x = self.data_x[s_begin:s_end]
        # print(f"self.data_x.shape: {self.data_x.shape}")
        # print(f"self.data_magnetogram.shape: {self.data_magnetogram.shape}")

        seq_magnetogram = self.data_magnetogram[s_begin:s_end]
        # print(f"index: {index}, seq_x: {seq_x.shape}, seq_magnetogram: {seq_magnetogram.shape}")

        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end] #2:28
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        # seq_y_mark = seq_y_mark[-1]

        
        return seq_x, seq_magnetogram, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols=self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns); cols.remove(self.target); cols.remove('Time')
        df_raw = df_raw[['Time']+cols+[self.target]]
        
        border1 = len(df_raw)-self.seq_len
        border2 = len(df_raw)
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        tmp_stamp = df_raw[['Time']][border1:border2]
        tmp_stamp['Time'] = pd.to_datetime(tmp_stamp["Time"], format='%Y-%m-%d %H:%M:%S')
        pred_dates = pd.date_range(tmp_stamp["Time"].values[-1], periods=self.pred_len+1, freq=self.freq)
        
        df_stamp = pd.DataFrame(columns = ['Time'])
        df_stamp['Time'] = list(tmp_stamp["Time"].values) + list(pred_dates[1:])
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq[-1:])

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index #0
        s_end = s_begin + self.seq_len #4
        r_begin = s_end - self.label_len #2
        r_end = r_begin + self.label_len + self.pred_len #28

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin+self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin+self.label_len] #2:4
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end] #2:28

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)