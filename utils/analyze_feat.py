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
from tools import StandardScaler
from timefeatures import time_features
import matplotlib.pyplot as plt

def analyze_feat(feat_path, target='logXmax1h', seq_len=1, set_type=0, scale=False, index=0):
    """
    Analyze the feature of the given path
    """
    df_raw = pd.read_csv(os.path.join(feat_path))

    
    # TODO check if name is correct

    '''
    df_raw.columns: ['Time', ...(other features), target feature]
    '''
    # cols = list(df_raw.columns); 

    cols = list(df_raw.columns)
    cols.remove(target)
    cols.remove('Time')
    df_raw = df_raw[['Time']+cols+[target]]

    # num_train = int(len(df_raw)*0.7)
    # num_test = int(len(df_raw)*0.2)
    num_train = 31439
    num_test = 8760
    
    num_vali = len(df_raw) - num_train - num_test
    # num_vali = 24 * 30 * 3

    print(f'df_raw[num_train]\n{df_raw[0:num_train]}')
    print(f'df_raw[num_train+num_test]\n{df_raw[num_train-seq_len:num_train+num_vali]}')
    print(f'df_raw[num_train+num_test+num_test]\n{df_raw[len(df_raw)-num_test-seq_len:len(df_raw)]}')

    # border1s = [0, num_train-self.seq_len, len(df_raw)-num_test-self.seq_len] #[train, val, test]
    # border2s = [num_train, num_train+num_vali, len(df_raw)] #[train, val, test]
    # border1s = [0, 0, 0]
    # border2s = [num_train, num_train, num_train]
    border1s = [0, num_train-seq_len, 0]
    border2s = [len(df_raw), num_train+num_vali, num_test]
    border1 = border1s[set_type]
    border2 = border2s[set_type]

    
    # TODO testをtrainを入れる

    cols_data = df_raw.columns[1:]
    df_data = df_raw[cols_data]
    data = df_data.values

    df_stamp = df_raw[['Time']][border1:border2]
    df_stamp['Time'] = pd.to_datetime(df_stamp["Time"], format='%Y-%m-%d %H:%M:%S')
    # data_stamp = time_features(df_stamp, timeenc=0, freq='h')
    
    data_x = data[border1:border2]
    # self.data_magnetogram = data_magnetogram[border1:border2]:
    data_y = data[border1:border2]

    span = 24 * 30 * 3
    s_begin = index # start index
    s_end = s_begin + span # end index
    r_begin = s_end # decoder start index, label_lenだけ前のデータを使う
    r_end = r_begin + 24

    df = df_raw.loc[border1+s_begin:border1 + s_end, ['Time', target]]
    df['Time'] = pd.to_datetime(df_stamp["Time"], format='%Y-%m-%d %H:%M:%S')
    df.set_index('Time', inplace=True)
    df.plot()


    # plt.title('logXmax1h')                            #グラフタイトル
    
    plt.ylabel('logXmax1h') #タテ軸のラベル
    plt.xlabel('date')                                #ヨコ軸のラベル
    # press any key to close
    plt.show()
    


if __name__ == '__main__':
    feat_path = 'data/data_all_stddev_fillna0.csv'

    for i in range(0, 24 * 30 * 12 * 8, 24 * 30 * 3):
        analyze_feat(feat_path, set_type=0, index=i)
        