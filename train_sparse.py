from mae.prod.train import *
from mae.prod.preprocess import *
import json
from src.Dataloader import TrainDatasetForPyramid, TrainDatasetSparse
from skimage.transform import resize

class FashionMnistDataLoader:
    def __init__(self):
        images, labels = self.load_mnist("/home/katsuyuki/temp/flare_transformer/fashion-mnist/data/fashion")
        self.images = torch.Tensor(images).unsqueeze(1)
        self.labels = torch.Tensor(labels)
        
        
    def __len__(self):
        """
        Returns:
            [int]: [length of sample]
        """
        print(self.images.shape)
        print(self.labels.shape)
        return len(self.images) // 60
        # return 130

    def __getitem__(self, idx):
        """
            get sample
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        #  fancy index

        sample = (self.images[idx],
                  self.labels[idx],
                  -1)

        return sample
        
    def calc_mean(self):
        """
            calculate mean and std of images
        """
        bs = 1000000000
        ndata = np.ravel(self.images)
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


    def load_mnist(self,path, kind='train'):
        import os
        import gzip
        import numpy as np

        """Load MNIST data from `path`"""
        labels_path = os.path.join(path,
                                '%s-labels-idx1-ubyte.gz'
                                % kind)
        images_path = os.path.join(path,
                                '%s-images-idx3-ubyte.gz'
                                % kind)

        with gzip.open(labels_path, 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                                offset=8)

        with gzip.open(images_path, 'rb') as imgpath:
            images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                                offset=16).reshape(len(labels), 28,28) / 255

        N,H,W = images.shape
        _images = np.empty((N,32,32))
        for i in range(N):
            _images[i,:,:] = resize(images[i,:,:],(32,32))
            # import cv2
            # x = np.empty((32,32,3))
            # for j in range(3): x[:,:,j] = _images[i,:,:]
            # # print(result.shape)
            # cv2.namedWindow('window')
            # cv2.imshow('window', x)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        return _images, labels

if __name__ == '__main__':
    # Preprocess().run()
     
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    params = json.loads(open("params/params_2014.json").read())
    train_dataset = TrainDatasetSparse("train", params["dataset"],has_window=False, grid_size=args.grid_size, keep_ratio=args.keep_ratio)

    val_dataset = TrainDatasetSparse("test", params["dataset"],has_window=False, grid_size=args.grid_size, keep_ratio=args.keep_ratio)

    # train_dataset = TrainDatasetForPyramid("train", params["dataset"],has_window=False, grid_size=args.grid_size)
    # val_dataset = TrainDatasetForPyramid("test", params["dataset"],has_window=False, grid_size=args.grid_size)
    # train_dataset = FashionMnistDataLoader()
    
    mean, std = train_dataset.calc_mean()
    print(mean, std)
    train_dataset.set_mean(mean, std)

    mean, std = val_dataset.calc_mean()
    print(mean, std)
    val_dataset.set_mean(mean, std)

    main(args,train_dataset, val_dataset)
    # main(args,train_dataset)
