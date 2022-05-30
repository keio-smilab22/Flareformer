import pandas as pd
import numpy as np
import json
import cv2
from datetime import datetime as dt
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import datetime
import glob

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
    # Extract columns other than ['logXmax1h', 'Xmax1h']
    feat_df = df.drop(['logXmax1h', 'Xmax1h'], axis=1)
    df_all = df.drop(['ID', 'Xhis', 'MHis', 'Xhis1d', 'Mhis1d', 'Chis', 'Chis1d', 'Class', 'X24', 'M24', 'XM24', 'C24', 'XMC24', 'Flrflag', 'Xmax1h', 'logXmax1m1', 'logXmax1m2', 'uv131max1h', 'uv131max1m1', 'uv131max1m2'], axis=1)

    # # fillna with -1.0
    # df["logXmax1h"] = df["logXmax1h"].fillna(-1.0000)
    # df["Xmax1h"] = df["Xmax1h"].fillna(1.0000e-7)

    # for col in df.columns:
    #     if col not in ['logXmax1h', 'Xmax1h']:
    #         df[col] = df[col].fillna(0.0)

    df_all.interpolate(method='bfill', inplace=True)
    # df_all.fillna(0.0, inplace=True)
    print(df_all.columns)

    # df_all.interpolate('time', inplace=True)

    xmax1h_df = df.loc[:,['logXmax1h', 'Xmax1h']]
    
    # # print(df.index.is_unique)
    save_path_target = 'data/data_target_v2.csv'
    save_path_feature = 'data/data_feat_v2.csv'
    save_path_all = 'data/data_all_v2_bfill.csv'
    # # save xmax1h_df to csv
    xmax1h_df.to_csv(save_path_target, index=True)
    feat_df.to_csv(save_path_feature, index=True)
    df_all.to_csv(save_path_all, index=True)


def data_plot(charval_path: str, path: str):
    df = pd.read_csv(path)
    df.rename(columns={'timdata-': 'Time'}, inplace=True)
    df['Time'] = pd.to_datetime(df['Time'], format='%d-%b-%Y %H:%M:%S.%f')
    df.set_index('Time', inplace=True)
    
    
    # print(len(df))

    # no header
    df_charval = pd.read_csv(charval_path, header=None)
    # print(df_charval.head(3))
    df.index = df.index.round('1H')
    df = df.resample('1H', label='left').max()
    print(df[:][:48961])
    print(df_charval[83][:])
    df['logXmax1h'][:48961].plot(figsize=(24, 6))
    plt.show()
    plt.savefig('data/logxmax1h_new.png')

    save_path_df = 'data/data_all_trial.csv'
    # df.to_csv(save_path_df, index=True)    
    

def data_gen_for_stddev(path:str):
    # df = pd.read_csv(path, index_col='Time', parse_dates=True, date_parser=lambda x: pd.to_datetime(x, format='%d-%b-%Y %H:%M:%S.%f'))

    df = pd.read_csv(path)
    
    # df.rename(columns={102: 'Time', 83:'logXmax1h'}, inplace=True)
    df.rename(columns={'timdata-': 'Time'}, inplace=True)
    print(df.columns)
    df.set_index('Time', inplace=True)
    df.index = pd.to_datetime(df.index, format='%d-%b-%Y %H:%M:%S.%f')
    df.index = df.index.round('1H')
    print(df.head(3))

    print(df.loc['2013-04-11'])
    df = df.resample('1H', label='left').max()
    print(df.columns)
    # Extract columns other than ['logXmax1h', 'Xmax1h']
    # df.drop(['logXmax1h', 'Xmax1h'], axis=1)
    df.drop(['Xmax1h', 'logXmax1m1', 'logXmax1m2', 'logXmax1p06', 'logXmax1p12', 'logXmax1p18', 'logXmax1p24'], axis=1, inplace=True)
    print(df.columns)
    print(df.loc['2017-04-11'])

    # df_stddev = pd.read_csv('data/stddev_charxxxall.csv')
    # df_stddev.rename(columns={'timdata-':'Time'}, inplace=True)
    # df_stddev['Time'] = pd.to_datetime(df_stddev['Time'], format='%d-%b-%Y %H:%M:%S.%f')
    # df_stddev.set_index('Time', inplace=True)
    
    # df.index = df.index.round('1H')
    # df_stddev = df_stddev.resample('1H', label='left').max()

    # merge
    # df_merge = pd.merge(df_stddev, df, how='left', left_index=True, right_index=True)
    # print(df_merge.loc[:, ['logxmax1h', 'logXmax1h']])

    method = 'ffill'
    df.interpolate(method=method, inplace=True)
    df.loc['2010-06-01 01:00:00'] = df.loc['2010-06-01 02:00:00']
    # df.fillna(0.0, inplace=True)
    print(df.loc['2010-06-01'])


    save_path_df = f'data/data_all_stddev_{method}.csv'
    df.to_csv(save_path_df, index=True)


    # # fillna with -1.0
    # df["logXmax1h"] = df["logXmax1h"].fillna(-1.0000)
    # df["Xmax1h"] = df["Xmax1h"].fillna(1.0000e-7)

    # for col in df.columns:
    #     if col not in ['logXmax1h', 'Xmax1h']:
    #         df[col] = df

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
            image_np_path = image_np_path.replace('../flare_transformer/data/magnetogram/', 'data/magnetogram/')
            if i < 10:
                print(image_np_path)

            image_np = cv2.imread(image_np_path)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
            image_np = cv2.resize(image_np, (256, 256))
            image_np = image_np[np.newaxis, :, :]
            # print(image_np.shape)
            image_data.append(image_np)
        image_data = np.stack(image_data, axis=0)
        print(image_data.shape)
        # save image_data to npy
        # np.save('data/data_magnetogram_256.npy', image_data)


def create_magnetogram_data_stddev(csv_path):
    df = pd.read_csv(csv_path, index_col='Time', parse_dates=True)
    
    image_data = []
    for d in tqdm(df.index, total=len(df.index), desc='Creating magnetogram data', dynamic_ncols=True):
        # print(d)
        # -2 minutues
        datetime_real = d - datetime.timedelta(minutes=2)
        year = datetime_real.year
        month = datetime_real.month
        day = datetime_real.day
        hour = datetime_real.hour
        minute = datetime_real.minute
        # second = datetime.second
        # microsecond = datetime.microsecond
        # print(year, month, day, hour, minute)
        dir_name = f'data/magnetogram/{year}-{month}'
        # example: data/magnetogram/2017-11/hmi.M_720s.20171120_085832.magnetogram.png
        
        file_path = glob.glob(os.path.join(dir_name, f'hmi.M_720s.{year}{month:02}{day:02}_{hour:02}????.magnetogram.png'))
        
        # file_path_start = 'data/magnetogram/2010-06/hmi.M_720s.20100601_005825.magnetogram.png'

        if len(file_path) != 0:
            print(file_path[0])
            image_np = cv2.imread(file_path[0])
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
            image_np = cv2.resize(image_np, (256, 256))
            image_np = image_np[np.newaxis, :, :]

            image_data.append(image_np)
            image_np_now = image_np
            # print(image_np.shape)
            

        else:
            print(f'{year}-{month}-{day} {hour}:{minute} is not found')
            if d == df.index[0]:
                image_np_now = cv2.imread('data/magnetogram/2010-06/hmi.M_720s.20100601_005825.magnetogram.png')
                image_np_now = cv2.cvtColor(image_np_now, cv2.COLOR_BGR2GRAY)
                image_np_now = cv2.resize(image_np_now, (256, 256))
                image_np_now = image_np_now[np.newaxis, :, :]
                image_data.append(image_np_now)
            else:
                image_data.append(image_np_now)

    image_data = np.stack(image_data, axis=0)
    print(image_data.shape)
    # save image_data to npy
    np.save('data/data_magnetogram_256_stddev.npy', image_data)
            


def main():
    # data_gen('data/defn_feature_database_v2.csv')
    # create_magnetogram_data('data/ft_database_all17.jsonl', 'data/defn_feature_database_v2.csv')
    # data_gen_new(charval_path='data/charval2017X_XMC24f_train_feat.csv', path='data/stddev_charxxxall.csv')
    # data_gen_new(charval_path='data/charval2017X_XMC24f_train_feat.csv', path='data/defn_feature_database_v2.csv')
    data_gen_for_stddev('data/defn_feature_database_stddev.csv')
    # create_magnetogram_data_stddev('data/data_all_stddev_fillna0.csv')





if __name__ == '__main__':
    main()