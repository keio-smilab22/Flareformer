from fileinput import filename
from turtle import color
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from  datetime import datetime
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame

import astropy.units as u
from astropy.table import Table
from astropy.time import Time, TimeDelta

import sunpy.data.sample
import sunpy.timeseries
from sunpy.net import Fido
from sunpy.net import attrs as a
from sunpy.time import TimeRange, parse_time
from sunpy.util.metadata import MetaDict

import seaborn as sns

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


def create_data_noaa_with_sunpy_ewm():
    # pd.options.display.float_format = '{:.7f}'.format
    year = 2010
    # goes = Fido.search(a.Time(f"{year}/01/01", f"{year}/12/31"), a.Instrument.xrs, a.goes.SatelliteNumber(15))
    goes = Fido.search(a.Time(f"{year}/01/01", f"{year}/12/31"), a.Instrument.xrs, a.goes.SatelliteNumber(15)|a.goes.SatelliteNumber(14)|a.goes.SatelliteNumber(13))

    print(goes)
    # goes = Fido.search(a.Time("2013/06/21", "2013/06/23"), a.Instrument.xrs)
    # show download file name
    goes_files = Fido.fetch(goes, max_conn=10, overwrite=False)
    while True:
        if len(goes_files.errors) == 0:
            break
        goes_files = Fido.fetch(goes_files, max_conn=100, overwrite=False)
    # goes.data
    print("files downloaded")
    # Using concatenate=True kwarg you can merge the files into one TimeSeries:
    combined_goes_ts = sunpy.timeseries.TimeSeries(goes_files, source='XRS', concatenate=True)
    print("files combined")
    
    df = combined_goes_ts.to_dataframe()
    print("dataframe created")

    # df = df[(df["xrsa_quality"] == 0) & (df["xrsb_quality"] == 0)]
    df['logxrsb'] = np.log10(df['xrsb'])
    df['logxmax1h'] = np.log10(df['xrsb']) + 6
    
    # print(combined_goes_ts.meta
    # print(combined_goes_ts.meta.to_string(100))
    # combined_goes_ts.plot(columns=["xrsb"])
    # df = combined_goes_ts.to_dataframe()
    # sns.histplot(df['logxmax1h'])
    # q = df['logxmax1h'].quantile(0.99998)
    # print(q)
    # Delete values greater than 10e-3
    # df = df[df['logxmax1h'] < 3]
    ewm_mean = df['logxmax1h'].ewm(span=1800).mean()
    ewm_std = df['logxmax1h'].ewm(span=1800).std()
    # df_ewm = df[(df['logxmax1h'] - ewm_mean).abs() < 3 * ewm_std]
    df_ewm = df
    print(len(df_ewm.loc[(df_ewm['logxmax1h'] - ewm_mean).abs() > 3 * ewm_std]))
    print(len(ewm_mean[(df['logxmax1h'] - ewm_mean).abs() > 3 * ewm_std]))
    # df_ewm.loc[(df_ewm['logxmax1h'] - ewm_mean).abs() > 3 * ewm_std, 'logxmax1h'] = ewm_mean[(df['logxmax1h'] - ewm_mean).abs() > 3 * ewm_std]
    
    # 訂正
    df_ewm.loc[(df_ewm['logxmax1h'] - ewm_mean).abs() > 3 * ewm_std, 'logxmax1h'] = np.nan
    fig, ax = plt.subplots()
    # log scale
    # ax.set_yscale('log')
    ax.set_ylim(-3.5, 3.5)
    df.plot(y="logxmax1h", ax=ax)
    ewm_mean.plot(color='r')
    
    df_downsampled = df_ewm.resample('60T', label='right', closed='right').max()
    # check missing data
    print(df_downsampled.isnull().sum())

    # fill missing values with previous value
    df_downsampled.fillna(method='ffill', inplace=True)
    print("missing values filled")

    # ewm_mean = df_downsampled['logxmax1h'].ewm(span=24).mean()
    # ewm_std = df_downsampled['logxmax1h'].ewm(span=24).std()
    # df_downsampled = df_downsampled[(df_downsampled['logxmax1h'] - ewm_mean).abs() < 3 * ewm_std]

    ts_downsampled = sunpy.timeseries.TimeSeries(df_downsampled,
                                             combined_goes_ts.meta,
                                             combined_goes_ts.units)
    fig, ax = plt.subplots()
    # log scale
    # ax.set_yscale('log')
    ax.set_ylim(-3.5, 3.5)
    ts_downsampled.plot(columns=["logxmax1h"])
    # ewm_mean.plot(color='r')
    # ts_downsampled.plot()
    # save to file
    ts_downsampled.to_dataframe().to_csv(os.path.join('data/noaa', f'xrs_downsampled_{year}.csv'))
    print(ts_downsampled.to_dataframe().info())

    # tr = TimeRange('2014-06-07 05:00', '2014-06-07 06:30')
    # ts_goes_trunc = combined_goes_ts.truncate(tr)
    # Or using strings:
    # ts_goes_trunc = combined_goes_ts.truncate('2014-06-07 05:00', '2014-06-07 06:30')
    # fig, ax = plt.subplots()
    # ts_goes_trunc.plot(columns=["xrsb"])
    plt.show()


def create_data_noaa_with_sunpy_new(year=2010):
    # pd.options.display.float_format = '{:.7f}'.format
    # pd.options.display.precision = 8

    
    # goes = Fido.search(a.Time(f"{year}/01/01", f"{year}/12/31"), a.Instrument.xrs, a.goes.SatelliteNumber(15))
    # goes = Fido.search(a.Time(f"{year}/01/01", f"{year}/12/31"), a.Instrument.xrs, a.goes.SatelliteNumber(15)|a.goes.SatelliteNumber(14)|a.goes.SatelliteNumber(13))
    goes = Fido.search(a.Time(f"{year}/01/01", f"{year}/12/31"), a.Instrument.xrs, a.goes.SatelliteNumber(15)|a.goes.SatelliteNumber(14)|a.goes.SatelliteNumber(13)|a.goes.SatelliteNumber(16)|a.goes.SatelliteNumber(17))

    print(goes)
    # goes = Fido.search(a.Time("2013/06/21", "2013/06/23"), a.Instrument.xrs)
    # show download file name
    goes_files = Fido.fetch(goes, max_conn=10, overwrite=False)
    while True:
        if len(goes_files.errors) == 0:
            break
        goes_files = Fido.fetch(goes_files, max_conn=100, overwrite=False)
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
    end_time = datetime(year, 12, 31)
    df = df[(df.index >= start_time) & (df.index <= end_time)]


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
    # fill missing values with previous value
    df.fillna(method='ffill', inplace=True)
    print(f"missing value\n{df.isna().sum()}")
    
    df['logxrsb'] = np.log10(df['xrsb'])
    df['logxmax1h'] = np.log10(df['xrsb']) + 6
    
    # if logxmax1h_t-1 - logxmax1h_t > 3, then logxmax1h_t = logxmax1h_t-1
    # df.loc[(df['logxmax1h'] - df['logxmax1h'].shift(freq='60T')).abs() > 3, 'logxmax1h'] = df['logxmax1h'].shift(freq='60T')

    # fig, ax = plt.subplots()
    # ax.set_ylim(-3.5, 3.5)
    # df.plot(y="logxmax1h", ax=ax)
    
    df_downsampled = df.resample('60T', label='left', closed='left').max()
    # two path
    # df_downsampled.loc[(df_downsampled['logxmax1h'] - df_downsampled['logxmax1h'].shift(1)).abs() > 3.0, 'logxmax1h'] = np.nan
    # df_downsampled.fillna(method='ffill', inplace=True)

    # one path
    for i in range(1, len(df_downsampled)):
        if (df_downsampled['logxmax1h'].iloc[i] - df_downsampled['logxmax1h'].iloc[i-1]) > 2.5:
            df_downsampled['logxmax1h'].iloc[i] = df_downsampled['logxmax1h'].iloc[i-1]

    # df_original_source = pd.read_csv("data/noaa/xrs_downsampled_2010_scale.csv")
    # df_original_source.set_index('time', inplace=True)
    # df_original_source.index = pd.to_datetime(df_original_source.index)
    # df_original_source.index = df_original_source.index.round('1H')
    # print(df_original_source.head())
    # print(df_original_source.index)
    

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
    
    # save to file
    ts_downsampled.to_dataframe().to_csv(os.path.join('data/noaa', f'xrs_downsampled_{year}.csv'))
    print(ts_downsampled.to_dataframe().info())
    
    # save to file
    plt.savefig(os.path.join(f'sunpy_logxmax1h_{year}.png'))
    
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
    for i in range(2010, 2022):
        create_data_noaa_with_sunpy_new(i)
    file_name = os.path.join(path_data, "xrs_downsampled_2012.csv")
    check_data(file_name=file_name, start_date=None, end_date=None)
    print("Data created.")
    