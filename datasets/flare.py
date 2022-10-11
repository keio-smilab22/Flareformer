"""Dataloader for Flare Transformer"""

from distutils.log import debug
import torch
import numpy as np
import os
from torch.utils.data import Dataset
from skimage.transform import resize
from tqdm import tqdm
from typing import Dict, Tuple, Any
from numpy import ndarray
from torch import Tensor
from numpy import float32


class FlareDataset(Dataset):
    def __init__(self,
                 dataset_type: str,
                 params: Dict[str, Any],
                 image_type: str = "magnetogram",
                 path: str = "data/data_",
                 debug: bool = False,
                 has_window: bool = True):

        print(f"====== {dataset_type} ======")
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
        print("Loading labels ...")
        self.label = self.get_multiple_year_data(start_year, end_year, "label")

        # get feat
        print("Loading features ...")
        self.feat = self.get_multiple_year_data(start_year, end_year, "feat")[:, :90]

        # get window
        print("Loading windows ...")
        self.window = self.get_multiple_year_window(start_year, end_year, "window_48")[:, :self.window_size]
        self.window = np.asarray(self.window, dtype=int)

        if debug:
            print(f"img: {self.img.shape}\n",
                  f"feat: {self.feat.shape}\n",
                  f"label: {self.label.shape}\n",
                  f"window: {self.window.shape}\n")
        else:
            shapes = [self.img.shape, self.feat.shape, self.label.shape, self.window.shape]
            samples = shapes[0][0]
            print(f"Samples: {samples}")
            assert all([samples == shape[0] for shape in shapes]), "The number of all samples must be equal."

    def __len__(self) -> int:
        """
        Returns:
            [int]: [length of sample]
        """
        return len(self.label)

    def __getitem__(self, idx: int) -> Tuple[Tuple[Tensor, ndarray], ndarray, int]:
        """
            get sample
        """
        if not self.has_window:
            return self.get_plain(idx)

        window = np.asarray(self.window[idx][:self.window_size], dtype=int)
        imgs, feats = self.img[window], self.feat[window]
        imgs = (imgs - self.mean) / self.std
        x = (imgs, feats)
        y = self.label[idx]

        return x, y, idx

    def get_plain(self, idx: int) -> Tuple[Tuple[Tensor, ndarray], ndarray, int]:
        imgs, feats = self.img[idx], self.feat[idx]
        imgs = (imgs - self.mean) / self.std
        x = (imgs, feats)
        y = self.label[idx]

        return x, y, idx

    def get_multiple_year_image(self, start_year: int, end_year: int, image_type: str) -> ndarray:
        """
            concatenate data of multiple years [image]
        """
        result = []
        for i, year in enumerate(tqdm(range(start_year, end_year + 1))):
            data_path_256 = f"{self.path}{year}_{image_type}_256.npy"
            data_path_512 = f"{self.path}{year}_{image_type}.npy"

            if os.path.exists(data_path_256):
                image_data = np.load(data_path_256)
            else:
                image_data = np.load(data_path_512)
                N, C, H, W = image_data.shape
                _image_data = np.empty((N, 1, 256, 256))
                for n in range(N):
                    source = image_data[n, 0, :, :].astype(np.uint8)
                    _image_data[n, 0, :, :] = resize(source, (256, 256))

                image_data = _image_data
                np.save(data_path_256, image_data)

            for j in range(image_data.shape[0]):
                assert np.max(image_data[j, :, :, :]) <= 1

            result.append(image_data)

        return np.concatenate(result, axis=0)

    def get_multiple_year_data(self, start_year: int, end_year: int, data_type: str) -> ndarray:
        """
            concatenate data of multiple years [feat/label]
        """
        result = []
        for year in tqdm(range(start_year, end_year + 1)):
            data_path = f"{self.path}{year}_{data_type}.csv"
            data = np.loadtxt(data_path, delimiter=',')
            result.append(data)
        return np.concatenate(result, axis=0)

    def get_multiple_year_window(self, start_year: int, end_year: int, data_type: str) -> ndarray:
        """
            concatenate data of multiple years [window]
        """
        result, N = [], 0
        for year in tqdm(range(start_year, end_year + 1)):
            data_path = f"{self.path}{year}_{data_type}.csv"
            data_path = self.path + str(year) + "_" + data_type + ".csv"
            data = np.loadtxt(data_path, delimiter=',')
            result.append(data + N)
            N += data.shape[0]
        return np.concatenate(result, axis=0)

    def calc_mean(self) -> Tuple[float32, float32]:
        """
            calculate mean and std of images
        """
        print("Calculate mean and std ...")
        bs = 1000000000
        ndata = np.ravel(self.img)
        mean = np.mean(ndata)
        std = 0
        for i in tqdm(range(ndata.shape[0] // bs + 1)):
            tmp = ndata[bs * i:bs * (i + 1)] - mean
            tmp = np.power(tmp, 2)
            std += np.sum(tmp)
        std = np.sqrt(std / len(ndata))
        print(f"(mean,std) = ({mean},{std})\n")

        return mean, std

    def set_mean(self, mean: float32, std: float32):
        """
            Set self.mean and self.std
        """
        self.mean = mean
        self.std = std


class OneshotDataset(Dataset):
    def __init__(self, imgs, feats, mean, std):
        self.imgs = [imgs]
        self.feats = [feats]
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> Tuple[Tuple[Tensor, ndarray], ndarray, int]:
        """
            get sample
        """

        imgs = (self.imgs[idx] - self.mean) / self.std
        x = (imgs, self.feats[idx])
        y_mock = np.array([1, 0, 0, 0])
        return x, y_mock, idx

class ForecastDataset(Dataset):
    def __init__(self,
                dataset_type: str,
                params: Dict[str, Any],
                image_type: str = "magnetogram",
                path: str = "data/data_",
                debug: bool = False,
                has_window: bool = True):

        print(f"====== {dataset_type} ======")
        self.path = path
        self.window_size = params["window"]
        self.has_window = has_window

        year_split = params["year_split"][dataset_type]
        start_year, end_year = year_split["start"], year_split["end"]

        # get x
        print("Loading images ...")
        self.img = self.get_multiple_year_image(start_year, end_year, image_type) # (48617, 1, 256, 256)
        self.img = torch.Tensor(self.img)

        # get label
        print("Loading labels ...")
        self.label = self.get_multiple_year_data(start_year, end_year, "label")

        # get feat
        print("Loading features ...")
        self.feat = self.get_multiple_year_data(start_year, end_year, "feat")[:, :90]

        # get window
        print("Loading windows ...")
        self.window = self.get_multiple_year_window(start_year, end_year, "window_forecast")[:, :self.window_size]
        # self.window = self.get_multiple_year_window(start_year, end_year, "window")[:, :self.window_size]
        self.window = np.asarray(self.window, dtype=int)

        # get forecast label
        print('Loading forecast ...')
        self.forecast = self.get_muliple_year_forcast(start_year, end_year, 'label')
        self.forecast = np.asarray(self.forecast, dtype=int)

        # get forecast label of active or not
        print('Loading forcast active or not ...')
        self.active = self.get_muliple_year_forcast(start_year, end_year, "active")
        self.active = np.asarray(self.active, dtype=int)

        if debug:
            print(f"img: {self.img.shape}\n",
                  f"feat: {self.feat.shape}\n",
                  f"label: {self.label.shape}\n",
                  f"window: {self.window.shape}\n")
        else:
            # shapes = [self.img.shape, self.feat.shape, self.label.shape, self.window.shape]
            shapes = [self.img.shape, self.feat.shape, self.label.shape]
            samples = shapes[0][0]
            print(f"Samples: {samples}")
            assert all([samples == shape[0] for shape in shapes]), "The number of all samples must be equal."

    def __len__(self) -> int:
        """
        Returns:
            [int]: [length of sample]
        """
        return len(self.window)

    def __getitem__(self, idx: int) -> Tuple[Tuple[Tensor, ndarray], ndarray, int]:
        """
            get sample
        """
        if not self.has_window:
            return self.get_plain(idx)

        window = np.asarray(self.window[idx][:self.window_size], dtype=int)
        imgs, feats = self.img[window], self.feat[window]
        imgs = (imgs - self.mean) / self.std
        forecast_class, active = self.forecast[idx], self.active[idx]
        x = (imgs, feats)
        y = (forecast_class, active)

        return x, y, idx

    def get_plain(self, idx: int) -> Tuple[Tuple[Tensor, ndarray], ndarray, int]:
        imgs, feats = self.img[idx], self.feat[idx]
        imgs = (imgs - self.mean) / self.std
        x = (imgs, feats)
        y = self.label[idx]

        return x, y, idx

    def get_multiple_year_image(self, start_year: int, end_year: int, image_type: str) -> ndarray:
        """
            concatenate data of multiple years [image]
        """
        result = []
        for i, year in enumerate(tqdm(range(start_year, end_year + 1))):
            data_path_256 = f"{self.path}{year}_{image_type}_256.npy"
            data_path_512 = f"{self.path}{year}_{image_type}.npy"

            if os.path.exists(data_path_256):
                image_data = np.load(data_path_256)
            else:
                image_data = np.load(data_path_512)
                N, C, H, W = image_data.shape
                _image_data = np.empty((N, 1, 256, 256))
                for n in range(N):
                    source = image_data[n, 0, :, :].astype(np.uint8)
                    _image_data[n, 0, :, :] = resize(source, (256, 256))

                image_data = _image_data
                np.save(data_path_256, image_data)

            for j in range(image_data.shape[0]):
                assert np.max(image_data[j, :, :, :]) <= 1

            result.append(image_data)

        return np.concatenate(result, axis=0)

    def get_multiple_year_data(self, start_year: int, end_year: int, data_type: str) -> ndarray:
        """
            concatenate data of multiple years [feat/label]
        """
        result = []
        for year in tqdm(range(start_year, end_year + 1)):
            data_path = f"{self.path}{year}_{data_type}.csv"
            data = np.loadtxt(data_path, delimiter=',')
            result.append(data)
        return np.concatenate(result, axis=0)

    def get_multiple_year_window(self, start_year: int, end_year: int, data_type: str) -> ndarray:
        """
            concatenate data of multiple years [window]
        """
        result, N = [], 0
        for year in tqdm(range(start_year, end_year + 1)):
            data_path = f"{self.path}{year}_{data_type}.csv"
            data = np.loadtxt(data_path, delimiter=',')
            result.append(data + N)
            N += data[-1][0]
        return np.concatenate(result, axis=0)

    def get_muliple_year_forcast(self, start_year: int, end_year: int, data_type: str) -> ndarray:
        """
            concatenate data of multiple years [template / DNN]
        """
        result = []
        for year in tqdm(range(start_year, end_year+1)):
            data_path = f"{self.path}{year}_forecast_{data_type}.csv"
            data = np.loadtxt(data_path, delimiter=',')
            result.append(data)
        return np.concatenate(result, axis=0)

    def calc_mean(self) -> Tuple[float32, float32]:
        """
            calculate mean and std of images
        """
        print("Calculate mean and std ...")
        bs = 1000000000
        ndata = np.ravel(self.img)
        mean = np.mean(ndata)
        std = 0
        for i in tqdm(range(ndata.shape[0] // bs + 1)):
            tmp = ndata[bs * i:bs * (i + 1)] - mean
            tmp = np.power(tmp, 2)
            std += np.sum(tmp)
        std = np.sqrt(std / len(ndata))
        print(f"(mean,std) = ({mean},{std})\n")

        return mean, std

    def set_mean(self, mean: float32, std: float32):
        """
            Set self.mean and self.std
        """
        self.mean = mean
        self.std = std