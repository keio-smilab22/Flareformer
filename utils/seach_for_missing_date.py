import datetime
import pandas as pd
import numpy as np
import argparse
import os
import glob
from tqdm import tqdm
import cv2
import glob
import shutil

def remove_image(dir_path):
    # if image size is not 1024x1024, remove it
    file_path = os.path.join(dir_path, '*.png')
    cnt = 0
    for file in tqdm(glob.glob(file_path)):
        img = cv2.imread(file, -1)
        print(img.shape)
        if img.shape != (1024, 1024, 4):
            # make new dir
            path_save = './data/noaa/magnetogram/irr/'
            if not os.path.exists(path_save):
                os.makedirs(path_save)
            # copy the file to a new directory
            new_file = os.path.join(path_save, os.path.basename(file))
            os.rename(file, new_file)

            # os.remove(file)
            print(f'{file} removed')
            cnt += 1
        else:
            print(f'{file} not removed')
    print(f'{cnt} files removed')


def seach_for_missing_date(csv_path):

    start_date = datetime(2011, 1, 1)
    end_date = datetime(2011, 2, 1)
    # Create dataframe that fits the 2011-01-01-00 format
    date_series = pd.date_range('2011-01-01', '2012-01-01', freq='H')
    # df
    df = pd.DataFrame(date_series, columns=['date'])
    # to date
    df['date'] = pd.to_datetime(df['date'])
    print(df)


    df_from_csv = pd.read_csv(csv_path, names=['filename'])
    # create new coulumn with the date
    # example : 
    # filename : hmi_m_45s_2011_05_01_00_01_30_tai_magnetogram.fits.png
    # date : 2011-05-01 00:01:30

    df_from_csv['date'] = df_from_csv['filename'].str.extract(r'(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})')
    df_from_csv['date'] = pd.to_datetime(df_from_csv['date'], format='%Y_%m_%d_%H_%M_%S')
    print(df_from_csv)
    # set index to date
    df_from_csv.set_index('date', inplace=True)
    print(df_from_csv)
    # resample to hourly
    df_from_csv = df_from_csv.resample('H').last()

    # merge with the original dataframe
    df = pd.merge(df, df_from_csv, on='date', how='outer')

    # fill missing values with NaN
    df = df.fillna(value=np.nan)
    
    print(df.head())
    print(df.tail())
    print(df.info())
    print(df.isnull().sum())

    # save to csv
    df.to_csv('/tmp/missing_date.csv', index=False)


if __name__ == '__main__':
    # argparse
    parser = argparse.ArgumentParser(description='Search for missing date in csv file')
    parser.add_argument('--csv_path', type=str, help='path to csv file', default='/tmp/2011_5m.csv', required=False)
    parser.add_argument('--dir_path', type=str, help='path to csv dir', default='./data/noaa/magnetogram/2011_5m/', required=False)
    args = parser.parse_args()
    csv_path = args.csv_path
    dir_path = args.dir_path
    # seach_for_missing_date(csv_path)

    # remove_image('./data/noaa/magnetogram/2011_5m')