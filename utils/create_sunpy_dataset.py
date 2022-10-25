import argparse
import datetime
import glob
import os
import shutil

import numpy as np
import pandas as pd
from tqdm import tqdm


def merge_logxmax1h_magnetogram_csv(path_logxmax1h_csv, path_magnetogram_csv, path_save):
    # read logxmax1h csv by pandas
    df_logxmax1h = pd.read_csv(path_logxmax1h_csv, index_col=0, parse_dates=True)
    df_logxmax1h = df_logxmax1h["logxmax1h"]
    print(df_logxmax1h.head())
    # extract logxmax1h date
    # df_logxmax1h[0] = pd.to_datetime(df_logxmax1h[0], format='%Y_%m_%d_%H_%M_%S')
    # # print(df_from_csv)
    # # set index to date
    # df_logxmax1h.set_index(0, inplace=True)

    # read magnetogram csv by pandas
    df_magnetogram = pd.read_csv(path_magnetogram_csv)
    df_magnetogram['date'] = pd.to_datetime(df_magnetogram['date'], format='%Y-%m-%d %H:%M:%S')

    # fill missing filename with filename of previous row
    df_magnetogram['filename'] = df_magnetogram['filename'].fillna(method='ffill')

    # merge two csv
    df_merged = pd.merge(df_magnetogram, df_logxmax1h, how='left', left_on='date', right_index=True)

    df_merged.rename(columns={'date': 'Time'}, inplace=True)

    # save merged csv
    df_merged.to_csv(path_save, index=False)


def concatenate_csv(end_year, path_save="./data/noaa/magnetogram_logxmax1h_all_years.csv"):
    # read merged csv by pandas
    for year in range(2011, end_year + 1):
        # get directory 
        save_dir = os.path.dirname(path_save)

        path_merged_csv = os.path.join(save_dir, f"magnetogram_logxmax1h_{year}.csv")
        df_merged = pd.read_csv(path_merged_csv)
        if year == 2011:
            df_concat = df_merged
        else:
            df_concat = pd.concat([df_concat, df_merged], axis=0)
    
    # save concatenated csv

    df_concat.to_csv(path_save, index=False)

def merge_sunpy_and_phys(path_sunpy_csv, path_phys_csv, path_save):
    # read sunpy csv by pandas
    df_sunpy = pd.read_csv(path_sunpy_csv, index_col=0, parse_dates=True)
    # extract sunpy date
    # df_sunpy[0] = pd.to_datetime(df_sunpy[0], format='%Y_%m_%d_%H_%M_%S')
    # # print(df_from_csv)
    # # set index to date
    # df_sunpy.set_index(0, inplace=True)

    # read phys csv by pandas
    df_phys = pd.read_csv(path_phys_csv)
    df_phys
    df_phys['Time'] = pd.to_datetime(df_phys['date'], format='%Y-%m-%d %H:%M:%S')

    # merge two csv
    df_merged = pd.merge(df_sunpy, df_phys, how='left', left_on='date', right_index=True)

    df_merged.rename(columns={'date': 'Time'}, inplace=True)

    # save merged csv
    df_merged.to_csv(path_save, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logxmax1h_csv', type=str, default='./data/noaa/xrs_downsampled_2018_no_process.csv')
    parser.add_argument('--magnetogram_csv', type=str, default='./data/noaa/magnetogram/2018/time_magnetogram_2018.csv')
    parser.add_argument('--save_csv', type=str, default='./data/noaa/magnetogram_logxmax1h_2018.csv')
    args = parser.parse_args()

    # merge_logxmax1h_magnetogram_csv(args.logxmax1h_csv, args.magnetogram_csv, args.save_csv)
    concatenate_csv(2014)
