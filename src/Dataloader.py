"""Dataloader for Flare Transformer"""

from statistics import mean
import torch
import torchvision.transforms as T
import numpy as np
import os
from torch.utils.data import Dataset
from skimage.transform import resize
import cv2
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
        self.img = self.get_multiple_year_image_dumm(year_split[split], image_type, grid_size, keep_ratio)
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
        # feat = self.feat[idx]
        # sample = ((x - self.mean) / self.std,
        #     self.label[idx],
        #     feat)
        # sample = (x,
        #     self.label[idx],
        #     feat)
        sample = (x, 0, 0)
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
            data_path_512 = self.path + str(year) + "_" + image_type + ".npy"
            # image_data = np.load(data_path_512)
            # print(np.max(image_data[0,0,:,:]))
            # print(np.max(resize(image_data[0,0,:,:],(256,256))))
                
            if not os.path.exists(data_path_sp):
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
                            chunks.append(col_img)
                            stds.append(std)
                chunks = np.array(chunks)
                stds = np.array(stds)
                ids = np.argsort(stds, axis=0)
                ids = ids[::-1]
                # keep the first subset
                ids_keep = ids[:len_keep]
                print(f"before chunks shape: {chunks.shape}") # (256*N, 1, 16, 16)
                chunks = chunks[ids_keep]
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


# class TrainDataloader(Dataset):
#     def __init__(self, split, params, image_type="magnetogram", path="data/data_", augmentation=False):
#         self.path = path
#         self.window_size = params["window"]
#         self.augmentation = augmentation

#         year_split = params["year_split"]

#         # get x
#         self.img = self.get_multiple_year_image(year_split[split], image_type)
#         # if self.augmentation:
#         #     transform = T.Compose([T.ToPILImage(),
#         #                             T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
#         #                             T.PILToTensor()])

#         #     S,C,H,W = self.img.shape
#         #     # img = torch.empty((S*2,C,H,W))
#         #     self.img = torch.Tensor(self.img)
#         #     img = torch.empty_like(self.img)
#         #     self.img = torch.cat((self.img,img),dim=0)
#         #     for i in range(S):
#         #         # img[i,:,:,:] = transform(self.img[i,0,:,:])
#         #         # img[i,:,:,:] = self.img[i,0,:,:]
#         #         # img[i+S,:,:,:] = transform(self.img[i,0,:,:])
#         #         self.img[i+S,:,:,:] = transform(self.img[i,0,:,:])

#         #     print(self.img.shape,img.shape)
#         #     print(self.img.shape)
        
#         # self.img = self.img[:, np.newaxis] # without fancy index

#         # get label
#         self.label = self.get_multiple_year_data(year_split[split], "label")
#         if self.augmentation: self.img = np.concatenate((self.label,self.label),axis=0)

#         # get feat
#         self.feat = self.get_multiple_year_data(year_split[split],
#                                                 "feat")[:, :90]

#         if self.augmentation: self.feat = np.concatenate((self.feat,self.feat),axis=0)

#         # get window
#         self.window = self.get_multiple_year_window(
#             year_split[split], "window_48")[:, :self.window_size]

#         self.window = np.asarray(self.window, dtype=int)
#         if self.augmentation: self.window = np.concatenate((self.window,self.window),axis=0)

#         print("img: {}".format(self.img.shape),
#               "label: {}".format(self.label.shape),
#               "feat: {}".format(self.feat.shape),
#               "window: {}".format(self.window.shape))

#     def __len__(self):
#         """
#         Returns:
#             [int]: [length of sample]
#         """
#         return 2 * len(self.label) if self.augmentation else len(self.label)

#     def __getitem__(self, idx):
#         """
#             get sample
#         """
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#         #  fancy index
#         over_table = idx >= len(self.label)
#         idx[over_table] -= len(self.label)
#         mul_x = self.img[np.asarray(self.window[idx][:self.window_size],
#                                     dtype=int)]
#         mul_feat = self.feat[np.asarray(self.window[idx][:self.window_size],
#                                         dtype=int)]
#         sample = ((mul_x - self.mean) / self.std,
#                   self.label[idx],
#                   mul_feat)

#         return sample

#     def get_multiple_year_image(self, year_dict, image_type):
#         """
#             concatenate data of multiple years [image]
#         """
#         for i, year in enumerate(range(year_dict["start"],
#                                  year_dict["end"]+1)):
#             data_path = self.path + str(year) + "_" + image_type + ".npy"
#             print(data_path)
#             image_data = np.load(data_path)
#             if i == 0:
#                 result = image_data
#             else:
#                 result = np.concatenate([result, image_data], axis=0)
#             # print(result.shape)
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
#         for i in range(ndata.shape[0] // bs + 1):
#             tmp = ndata[bs*i:bs*(i+1)] - mean
#             tmp = np.power(tmp, 2)
#             std += np.sum(tmp)
#             print("Calculating std : ", i, "/", ndata.shape[0] // bs)
#         std = np.sqrt(std / len(ndata))
#         return mean, std

#     def set_mean(self, mean, std):
#         """
#             Set self.mean and self.std
#         """
#         self.mean = mean
#         self.std = std
