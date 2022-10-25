import argparse
import datetime
import glob
import os
import shutil

import numpy as np
import pandas as pd
from tqdm import tqdm


def seach_for_representative_image(path_dir, year=2011):
    
    # for the leap year
    if year % 4 == 0:
        month_span = {1:31, 2:29, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}
    else:
        month_span = {1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}

    for month in range(1, 13):
        start_date = datetime.datetime(year, month, 1)
        # 1 month after start_date
        end_date = start_date + datetime.timedelta(days=month_span[month]-1)

        # Create dataframe that fits the 2011-01-01-00 format
        date_series = pd.date_range(start_date, end_date, freq='H')
        # df
        df = pd.DataFrame(date_series, columns=['date'])
        # to date
        df['date'] = pd.to_datetime(df['date'])


        # path_csv = os.path.join(path_dir, f'{year}_5m_{month}.csv')

        # get filenames with glob
        file_path = os.path.join(path_dir, f'{month:02d}', f'hmi_m_45s_{year}_{month:02d}_*.png')
        file_list = glob.glob(file_path)
        # print(file_list)
        # only keep the filename
        file_list = [os.path.basename(file) for file in file_list]
        # create new dataframe with filenames
        df_from_csv = pd.DataFrame(file_list, columns=['filename'])
        # print(df_from_csv)

        # df_from_csv = pd.read_csv(path_csv, names=['filename'])
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
        df_from_csv = df_from_csv.resample('H').first()
        print(df_from_csv)

        # copy the file to a new directory
        path_origin = os.path.join(path_dir, f'{month:02d}')
        # path_save_dir is upper directory of path_origin_dir   
        path_save_dir = os.path.dirname(path_dir)
        path_save = os.path.join(path_save_dir, f'{year}', f'{month:02d}')

        if not os.path.exists(path_save):
            os.makedirs(path_save)
        # copy the file to a new directory
        for file in df_from_csv['filename']:
            # file is None if the file is not found
            if file is not None:
                file = os.path.join(path_origin, os.path.basename(file))
                new_file = os.path.join(path_save, os.path.basename(file))
                
                if not os.path.exists(new_file):
                    shutil.copy(file, new_file)
                # print(f'{file} copied')
        

    # merge with the original dataframe
    df = pd.merge(df, df_from_csv, on='date', how='outer')

    # fill missing values with NaN
    df = df.fillna(value=np.nan)
    
    print(df.head())
    print(df.tail())
    print(df.info())
    print(df.isnull().sum())

    # save to csv
    # path_save_csv = os.path.join(path_save, 'representative_image.csv')
    df.to_csv('/tmp/missing_date.csv', index=False)


def create_magnetogram_csv(path_dir, year=2011):
    # for the leap year
    if year % 4 == 0:
        month_span = {1:31, 2:29, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}
    else:
        month_span = {1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}

    df_all = pd.DataFrame()

    for month in range(1, 13):
        start_date = datetime.datetime(year, month, 1)
        # 1 month after start_date
        end_date = start_date + datetime.timedelta(days=month_span[month]-1)

        # Create dataframe that fits the 2011-01-01-00 format
        date_series = pd.date_range(start_date, end_date, freq='H')
        # df
        df = pd.DataFrame(date_series, columns=['date'])
        # to date
        df['date'] = pd.to_datetime(df['date'])


        # path_csv = os.path.join(path_dir, f'{year}_5m_{month}.csv')

        # get filenames with glob
        file_path = os.path.join(path_dir, f'{month:02d}', f'hmi_m_45s_{year}_{month:02d}_*.png')
        file_list = glob.glob(file_path)
        # print(file_list)
        # only keep the filename
        file_list = [os.path.basename(file) for file in file_list]
        # create new dataframe with filenames
        df_from_csv = pd.DataFrame(file_list, columns=['filename'])
        # print(df_from_csv)

        # df_from_csv = pd.read_csv(path_csv, names=['filename'])
        # create new coulumn with the date
        # example : 
        # filename : hmi_m_45s_2011_05_01_00_01_30_tai_magnetogram.fits.png
        # date : 2011-05-01 00:01:30

        df_from_csv['date'] = df_from_csv['filename'].str.extract(r'(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})')
        df_from_csv['date'] = pd.to_datetime(df_from_csv['date'], format='%Y_%m_%d_%H_%M_%S')
        # print(df_from_csv)
        # set index to date
        df_from_csv.set_index('date', inplace=True)
        # print(df_from_csv)
        # resample to hourly
        df_from_csv = df_from_csv.resample('H').first()
        # print(df_from_csv)

        # merge with the original dataframe
        df = pd.merge(df, df_from_csv, on='date', how='outer')

        # fill missing values with NaN
        df = df.fillna(value=np.nan)
        
        print(df.head())
        print(df.tail())
        print(df.info())
        print(df.isnull().sum())

        # add to the all dataframe
        df_all = df_all.append(df)
    print(df_all.head())
    print(df_all.isnull().sum())
    

    # save to csv
    path_save_csv = os.path.join(path_dir, f'time_magnetogram_{year}.csv')
    df_all.to_csv(path_save_csv, index=False)
    print(f'{path_save_csv} saved')


if __name__ == '__main__':
    # argparse
    parser = argparse.ArgumentParser(description='Select magenetogram in csv file')
    parser.add_argument('--csv_path', type=str, help='path to csv file', default='./data/noaa/magnetogram/2013/time_magnetogram_2013.csv', required=False)
    parser.add_argument('--dir_path', type=str, help='path to csv dir', default='./data/noaa/magnetogram/2013', required=False)
    parser.add_argument('--year', type=int, help='year', default='2013', required=True)
    args = parser.parse_args()

    dir_path = args.dir_path
    year = args.year
    # seach_for_representative_image(dir_path, year)
    create_magnetogram_csv(dir_path, year)
