import mae.prod.train as magnetogram
import mae.prod.train_seq_mae as seq
import mae.prod.train_phy as physics
from mae.prod.preprocess import *
import json
import argparse
from pathlib import Path 
from src.Dataloader import TrainDataloader256
from skimage.transform import resize

class FashionMnistDataLoader:
    def __init__(self):
        images, labels = self.load_mnist("/home/initial/workspace/flare_transformer/fashion-mnist/data/fashion")
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

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument(
        '--batch_size',
        default=130,
        type=int,
        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--dim', default=512, type=int)
    parser.add_argument('--baseline', default="attn")
    parser.add_argument('--enc_depth', default=12, type=int)
    parser.add_argument('--dec_depth', default=8, type=int)
    parser.add_argument(
        '--accum_iter',
        default=1,
        type=int,
        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    parser.add_argument('--input_size', default=256, type=int,  # 512の場合はここ変える
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument(
        '--model',
        default='vit_for_FT',
        type=str,
        metavar='MODEL',
        help='Name of model to train')

    parser.add_argument(
        '--norm_pix_loss',
        action='store_true',
        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=0.0002, metavar='LR',  # 0.0000007
                        help='learning rate (absolute lr)')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N', # warmup: [0,5]
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--output_dir', default='../../workspace/flare_transformer/output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument(
        '--pin_mem',
        action='store_true',
        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--batch_size_search', default=False, action='store_true')

    parser.add_argument('--target', default="m") # m, p, seq
    parser.add_argument('--token_window', default=4, type=int)
    parser.set_defaults(pin_mem=True)

    return parser


def get_train_dataset(window=4, has_window=False):
    params = json.loads(open("params/params_2014.json").read())
    params["dataset"]["window"] = window
    train_dataset = TrainDataloader256("train", params["dataset"],has_window=has_window)
    
    mean, std = train_dataset.calc_mean()
    print(mean, std)
    train_dataset.set_mean(mean, std)
    return train_dataset, mean, std

def get_test_dataset(mean,std,window=4, has_window=False):
    params = json.loads(open("params/params_2014.json").read())
    params["dataset"]["window"] = window
    test = TrainDataloader256("test", params["dataset"],has_window=has_window)
    test.set_mean(mean, std)
    return test


if __name__ == '__main__':
    # Preprocess().run()
     
    parser = get_args_parser()
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.target == "m":
        train_dataset, mean, std = get_train_dataset(has_window=False)
        magnetogram.main(args,train_dataset)
    elif args.target == "seq":
        train_dataset, mean, std = get_train_dataset(window=2, has_window=True)
        test_dataset = get_test_dataset(mean,std,window=2,has_window=True)
        seq.main(args,train_dataset,test_dataset)
    else:
        train_dataset, mean, std = get_train_dataset(window=12, has_window=True)
        physics.main(args,train_dataset)
