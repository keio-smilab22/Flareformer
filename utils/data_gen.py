import pandas as pd
import numpy as np
import json
import cv2
from datetime import datetime as dt
from tqdm import tqdm


def data_gen(path: str):
    df = pd.read_csv(path, index_col='Time', parse_dates=True, date_parser=lambda x: pd.to_datetime(x, format='%d-%b-%Y %H:%M:%S.%f'))
    # print(df.head(3))
    # print(len(df))
    # print(len(df['Time'].unique()))
    # df.set_index('Time', inplace=True)
    # df['Time'] = pd.to_datetime(df['Time'], format='%d-%b-%Y %H:%M:%S.%f')
    # print(len(df))
    df.index = df.index.round('1H')
    
    # print(df.loc['2011-11-11'])
    df = df.resample('1H', label='left').max()
    
    # fillna with -1.0
    df["logXmax1h"] = df["logXmax1h"].fillna(-1.0000)
    df["Xmax1h"] = df["Xmax1h"].fillna(1.0000e-7)

    for col in df.columns:
        if col not in ['logXmax1h', 'Xmax1h']:
            df[col] = df[col].fillna(0.0)

    print(df.columns)


    xmax1h_df = df.loc[:,['logXmax1h', 'Xmax1h']]
    # Extract columns other than ['logXmax1h', 'Xmax1h']
    feat_df = df.drop(['logXmax1h', 'Xmax1h'], axis=1)
    df_all = df.drop(['ID', 'Xhis', 'MHis', 'Xhis1d', 'Mhis1d', 'Chis', 'Chis1d', 'Class', 'X24', 'M24', 'XM24', 'C24', 'XMC24', 'Flrflag', 'Xmax1h', 'logXmax1m1', 'logXmax1m2', 'uv131max1h', 'uv131max1m2'], axis=1)
    # # print(df.index.is_unique)
    save_path_target = '../data/data_target_v2.csv'
    save_path_feature = '../data/data_feat_v2.csv'
    save_path_all = '../data/data_all_v2.csv'
    # # save xmax1h_df to csv
    xmax1h_df.to_csv(save_path_target, index=True)
    feat_df.to_csv(save_path_feature, index=True)
    df_all.to_csv(save_path_all, index=True)


def create_magnetogram_data(jsonl_path: str, csv_path):
    df = pd.read_csv(csv_path, index_col='Time', parse_dates=True, date_parser=lambda x: pd.to_datetime(x, format='%d-%b-%Y %H:%M:%S.%f'))
    df.index = df.index.round('1H')
    df = df.resample('1H', label='left').max()

    with open(jsonl_path, 'r') as f:
        data = [json.loads(line) for line in f]

        image_data = []
        for i in tqdm(range(len(data)), total=len(data), desc='Creating magnetogram data', dynamic_ncols=True):
            # time : 01-Jun-2010 02
            time = dt.strptime(data[i]['time'], '%d-%b-%Y %H')
            image_np_path = data[i]['magnetogram']
            # image_np_path = '../flare_transformer/data/magnetogram/2017-11/hmi.M_720s.20171120_085832.magnetogram.png'
            # image_np_path -> '../data/magnetogram/2017-11/hmi.M_720s.20171120_085832.magnetogram.png'
            image_np_path = image_np_path.replace('../flare_transformer/data/magnetogram/', '/home/katsuyuki/temp/flare_transformer/data/magnetogram/')
            # print(image_np_path)
            image_np = cv2.imread(image_np_path)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
            image_np = cv2.resize(image_np, (256, 256))
            image_np = image_np[np.newaxis, :, :]
            print(image_np.shape)
            image_data.append(image_np)
        image_data = np.stack(image_data, axis=0)
        print(image_data.shape)
        # save image_data to npy
        np.save('/home/katsuyuki/temp/flare_transformer/data/data_magnetogram_256.npy', image_data)







def main():
    data_gen('../data/defn_feature_database_v2.csv')
    # create_magnetogram_data('data/ft_database_all17.jsonl', 'data/defn_feature_database_v2.csv')


if __name__ == '__main__':
    main()