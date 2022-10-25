import os
from collections import OrderedDict
from datetime import datetime
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sunpy.data.sample
import sunpy.timeseries
from astropy.table import Table
from astropy.time import Time, TimeDelta
from pandas import DataFrame
from sunpy.net import Fido
from sunpy.net import attrs as a
from sunpy.time import TimeRange, parse_time
from sunpy.util.metadata import MetaDict


def preprocess_data_noaa(path_d):
    # Find the part of g14_xrs_2s_2010{month:02d}{i:02d}_2010{month:02d}{i:02d}.csv that says "data" and delete the line before it.
    
    for month in range(1, 13):
        for i in range(1, 32):
            
            if os.path.exists(os.path.join(path_d, f'g14_xrs_2s_2010{month:02d}{i:02d}_2010{month:02d}{i:02d}.csv')):
                file_path = os.path.join(path_d, f'g14_xrs_2s_2010{month:02d}{i:02d}_2010{month:02d}{i:02d}.csv')
            else:
                file_path = os.path.join(path_d, f'g15_xrs_2s_2010{month:02d}{i:02d}_2010{month:02d}{i:02d}.csv')
                if not os.path.exists(file_path):
                        print(f'File {file_path} does not exist')
                        break
            with open(file_path, 'r') as f:
                lines = f.readlines()
                for idx, line in enumerate(lines):
                    if line.startswith('data:'):
                        print(idx)



def create_data_noaa(path_d, path_save):
    """
    NOTE: Actual data starts at line 140 of the csv file.

    """
    # Reads daily csv files and concatenates them at the end
    
    # path_d = 'data/noaa/'
    # path_save = 'data/noaa/'
    for month in range(1, 13):
        for i in range(1, 32):
            
            if os.path.exists(os.path.join(path_d, f'g14_xrs_2s_2010{month:02d}{i:02d}_2010{month:02d}{i:02d}.csv')):
                file_path = os.path.join(path_d, f'g14_xrs_2s_2010{month:02d}{i:02d}_2010{month:02d}{i:02d}.csv')
            else:
                file_path = os.path.join(path_d, f'g15_xrs_2s_2010{month:02d}{i:02d}_2010{month:02d}{i:02d}.csv')
                if not os.path.exists(file_path):
                        print(f'File {file_path} does not exist')
                        break
            skip_lines = 0
            with open(file_path, 'r') as f:
                lines = f.readlines()
                for idx, line in enumerate(lines):
                    if line.startswith('data:'):
                        print(idx)
                        skip_lines = idx
            df = pd.read_csv(file_path, skiprows=skip_lines+1, index_col=0, parse_dates=True)
            print(df.info())
            if i == 1 and month == 1:
                df_all = df
            else:
                df_all = pd.concat([df_all, df])
    df_downsampled = df_all.resample('60T', label='right').max()
    # add new column logxmax1h
    df_downsampled['logxmax1h'] = np.log10(df_downsampled['B_FLUX']) + 6
    print(df_downsampled.info())
    # save to file
    df_downsampled.to_csv(path_save)
    # plot the data
    # plot the data column logxmax1h

    df_downsampled.plot(y='logxmax1h')
    # df_downsampled.plot()

    plt.show()




def create_data_noaa_with_sunpy_new(year=2010, thr=3):
    # pd.options.display.float_format = '{:.7f}'.format
    # pd.options.display.precision = 8

    
    # goes = Fido.search(a.Time(f"{year}/01/01", f"{year}/12/31"), a.Instrument.xrs, a.goes.SatelliteNumber(15))
    goes = Fido.search(a.Time(f"{year}/01/01", f"{year+1}/01/01"), a.Instrument.xrs, a.goes.SatelliteNumber(15)|a.goes.SatelliteNumber(14)|a.goes.SatelliteNumber(13))
    # goes = Fido.search(a.Time(f"{year}/10/01", f"{year}/10/31"), a.Instrument.xrs, a.goes.SatelliteNumber(15)|a.goes.SatelliteNumber(14)|a.goes.SatelliteNumber(13)|a.goes.SatelliteNumber(16)|a.goes.SatelliteNumber(17))

    print(goes)
    goes_files = Fido.fetch(goes, max_conn=10, overwrite=False)
    while True:
        if len(goes_files.errors) == 0:
            break
        goes_files = Fido.fetch(goes_files, max_conn=10, overwrite=False)
    # goes.data
    print("files downloaded")
    # Using concatenate=True kwarg you can merge the files into one TimeSeries:
    combined_goes_ts = sunpy.timeseries.TimeSeries(goes_files, source='XRS', concatenate=True)
    print("files combined")
    
    df = combined_goes_ts.to_dataframe()

    # drop xrsa column
    df.drop(columns=['xrsa'], inplace=True)

    # Set the negative value to 10e-9
    df.loc[df['xrsb'] <= 0, 'xrsb'] = np.nan
    # only keep {year}/01/01 to {year}/12/31
    start_time = datetime(year, 1, 1)
    end_time = datetime(year+1, 1, 1)
    df = df[(df.index >= start_time) & (df.index <= end_time)]

    print(df.tail())


    df.index = df.index.round('S')
    # find duplicated index
    print(df.index.duplicated().sum())
    # delete duplicated index and keep the biggest value
    df = df.resample('2S', label='left', closed='left').max()
    print(df.index.duplicated().sum())
    df = df.asfreq('2S', fill_value=np.nan)

    # df_nan = df[df['xrsb'].isnull()]
    # print(df_nan.head())
    print("dataframe created")
    
    df_downsampled:pd.DataFrame = df.resample('60T', label='left', closed='left').max()
    # 欠損値が連続している箇所毎にグループ番号を振る
    df_downsampled['nan_group'] = ((df_downsampled['xrsb'].isna()) & (df_downsampled['xrsb'].shift(1).notna())).where(df_downsampled['xrsb'].isna()).cumsum()
    # 上記のグループ毎に欠損値が何個連続しているかを求める
    df_downsampled['nan_count'] = df_downsampled['nan_group'].map(df_downsampled.groupby('nan_group').size())

    print(df_downsampled[df_downsampled['nan_count'].notna()])
    # 箇所毎にグループ番号はもう使わないので削除
    df_downsampled = df_downsampled.drop(columns=['nan_group'])

    # 欠損値がN個連続している箇所以外を補間
    N=24
    
    df_downsampled['xrsb'] = df_downsampled.loc[(df_downsampled['nan_count'] < N) | (df_downsampled['nan_count'].isna()), 'xrsb'].interpolate(method='ffill')
    df_downsampled = df_downsampled.drop(columns=['nan_count'])
    # fill missing values with previous value
    # df.fillna(method='ffill', inplace=True)
    # if xrsb is nan, then drop xrsb
    # df_downsampled.dropna(subset=['xrsb'], inplace=True)
    
    df_downsampled['logxrsb'] = np.log10(df_downsampled['xrsb'])
    df_downsampled['logxmax1h'] = np.log10(df_downsampled['xrsb']) + 6
    print("dataframe processed")
    print(df_downsampled)

    # two path
    # df_downsampled.loc[(df_downsampled['logxmax1h'] - df_downsampled['logxmax1h'].shift(1)).abs() > 3.0, 'logxmax1h'] = np.nan
    # df_downsampled.fillna(method='ffill', inplace=True)

    # one path
    # for i in range(1, len(df_downsampled)):
    #     if (df_downsampled['logxmax1h'].iloc[i] - df_downsampled['logxmax1h'].iloc[i-1]) > thr:
    #         df_downsampled['logxmax1h'].iloc[i] = df_downsampled['logxmax1h'].iloc[i-1]

    # If the value is the same for 24 consecutive hours, delete the row
    # df_downsampled_copy = df_downsampled.copy()
    # for i in range(1, len(df_downsampled)-24):
    #     print(i)
    #     if df_downsampled['logxmax1h'].iloc[i] == df_downsampled['logxmax1h'].iloc[i-1]:
    #         cnt = 0
    #         for j in range(i, i+24):
    #             if df_downsampled['logxmax1h'].iloc[j] == df_downsampled['logxmax1h'].iloc[i]:
    #                 cnt += 1
    #         if cnt == 24:
    #             df_downsampled_copy.drop(df_downsampled.index[i])


    

    # print(df_downsampled.isnull().sum())

    # fill missing values with previous value
    # df_downsampled.fillna(method='ffill', inplace=True)
    # print("missing values filled")

    ts_downsampled = sunpy.timeseries.TimeSeries(df_downsampled,
                                             combined_goes_ts.meta,
                                             combined_goes_ts.units)
    fig, ax = plt.subplots()
    ax.set_ylim(-2.5, 3.5)
    ts_downsampled.plot(columns=["logxmax1h"])
    # df_downsampled.plot(y="logxmax1h", ax=ax)
    
    # save to file
    ts_downsampled = ts_downsampled.to_dataframe()
    ts_downsampled.dropna(subset=['xrsb', 'logxrsb', 'logxmax1h'], inplace=True)

    # ts_downsampled['xrsb'] = ts_downsampled['xrsb'].map(lambda x: float(Decimal(str(x)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)))
    # ts_downsampled['logxrsb'] = ts_downsampled['logxrsb'].map(lambda x: float(Decimal(str(x)).quantize(Decimal('0.000001'), rounding=ROUND_HALF_EVEN)))
    # ts_downsampled['logxmax1h'] = ts_downsampled['logxmax1h'].map(lambda x: float(Decimal(str(x)).quantize(Decimal('0.000001'), rounding=ROUND_HALF_EVEN)))
    ts_downsampled['logxrsb'] = ts_downsampled['logxrsb'].map('{:.6f}'.format)
    ts_downsampled['logxmax1h'] = ts_downsampled['logxmax1h'].map('{:.6f}'.format)
    
    ts_downsampled.to_csv(os.path.join('data/noaa', f'xrs_downsampled_{year}_no_process.csv'))
    print(ts_downsampled.info())
    
    # save to file
    # plt.savefig(os.path.join(f'sunpy_logxmax1h_{year}_{thr}.png'))
    plt.savefig(os.path.join("images", f'sunpy_logxmax1h_{year}_no_process.png'))
    # plt.show()


def check_data(file_name, start_date:str = None, end_date:str = None):
    df = pd.read_csv(file_name, index_col=0, parse_dates=True)
    if start_date is not None:
        start_date = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
        df = df.loc[df.index >= start_date]
    if end_date is not None:
        end_date = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S")
        df = df.loc[df.index <= end_date]
    
    print(df.info())
    print(df.head())
    print(df.tail())
    print(df.describe())
    print(df.isnull().sum())
    
    df.plot(y="logxmax1h")
    plt.ylim(-2.5, 3.5)
    plt.show()

if __name__ == "__main__":
    path_data = os.path.join("data/noaa/")
    path_save = os.path.join("data/noaa/g_15_new.csv")
    # create_data_noaa(path_data, path_save)
    # create_data_original_noaa("/tmp/tmp_2010.csv")
    for thr in [3.0, 2.5]:
        for i in range(2010, 2022):
            create_data_noaa_with_sunpy_new(year=i, thr=thr)
    # create_data_noaa_with_sunpy_new(year=2014, thr=0)
    
    # file_name = os.path.join(path_data, f"xrs_downsampled_2012_{thr}.csv")
    # check_data(file_name=file_name, start_date=None, end_date=None)
    print("Data created.")
    