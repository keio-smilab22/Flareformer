import argparse
from argparse import Namespace
from time import sleep

import numpy as np
import ray
from datasets import detect_year_sections, get_image, read_jsonl, split_dataset
from utils import identity, make_prefix, sub_1h
from tqdm import tqdm

import pandas as pd
import datetime
import json


def replace_sunpy_image(dataset_path: str, csv_path: str, output_path: str):
    """
    Replace sunpy image
    """
    print("Load dataset ... ")
    data = read_jsonl(dataset_path)


    print("Load csv ... ")
    sunpy = pd.read_csv(csv_path)

    print("Replace sunpy image ... ")
    data_new = data.copy()

    remove_index = []

    for i, d in tqdm(enumerate(data), total=len(data)):
        data_time = data_new[i]["time"]
        data_time = datetime.datetime.strptime(data_time, "%d-%b-%Y %H")
        data_time = data_time.replace(minute=0, second=0, microsecond=0)

        if data_time < datetime.datetime.strptime("01-Jan-2011 00", "%d-%b-%Y %H"):
            # # remove i data
            # remove_index.append(i)
            continue
        elif data_time < datetime.datetime.strptime("01-Jan-2018 00", "%d-%b-%Y %H") and data_time >= datetime.datetime.strptime("01-Jan-2011 00", "%d-%b-%Y %H"):
            # print("skip", data_time)
            # replace sunpy data
            # get row index of sunpy["Time"] which is same as data_time
            sunpy_index = sunpy[sunpy["Time"] == data_time.strftime("%Y-%m-%d %H:%M:%S")].index
            if len(sunpy_index) == 0:
                remove_index.append(i)
                continue
                
            # filename : hmi_m_45s_2011_01_01_00_01_30_tai_magnetogram.fits.png
            year = int(sunpy["filename"].iloc[sunpy_index[0]].split("_")[3])
            month = int(sunpy["filename"].iloc[sunpy_index[0]].split("_")[4])
            data_new[i]["magnetogram"] = f"../flare_transformer/data/noaa/magnetogram/{year}/{month:02d}/{sunpy['filename'].iloc[sunpy_index[0]]}"

        else:
            # replace sunpy data
            # get row index of sunpy["Time"] which is same as data_time
            sunpy_index = sunpy[sunpy["Time"] == data_time.strftime("%Y-%m-%d %H:%M:%S")].index
            if len(sunpy_index) == 0:
                remove_index.append(i)
                continue

            # filename : hmi_m_45s_2011_01_01_00_01_30_tai_magnetogram.fits.png
            year = int(sunpy["filename"].iloc[sunpy_index[0]].split("_")[3])
            month = int(sunpy["filename"].iloc[sunpy_index[0]].split("_")[4])
            
            # data[i]["magnetogram"] = f"../flare_transformer/data/noaa/magnetogram/{year}/{month:02d}/{sunpy['filename'].iloc[sunpy_index[0]]}"
            data_new[i]["magnetogram"] = f"../flare_transformer/data/noaa/magnetogram/{year}/{month:02d}/{sunpy['filename'].iloc[sunpy_index[0]]}"
            # calculate flag
            flag = [0, 0, 0, 0]
            max_logxmax1h = max(sunpy["logxmax1h"].iloc[sunpy_index[0]:sunpy_index[0]+24])

            flag = ",".join([str(f) for f in flag])

            data_new[i]["flag"] = flag
            data_new[i]["feature"] = "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"
    # remove data
    data_new = [d for i, d in enumerate(data_new) if i not in remove_index]
    
    print("Add new flag ... ")
    # remove nan data
    before_len = len(sunpy)
    sunpy = sunpy.dropna(subset=["logxmax1h"])
    after_len = len(sunpy)
    print(f"remove {before_len - after_len} nan data")

    data = read_jsonl(dataset_path)

    none_skip_count = 0
    for i, s in enumerate(sunpy["Time"]):

        date_time = datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        date_time = date_time.replace(minute=0, second=0, microsecond=0)

        # before "24-Dec-2017 12" skip
        if date_time < datetime.datetime.strptime("24-Dec-2017 13", "%d-%b-%Y %H"):
            # print("skip", data_time)
            continue
        date_time = date_time.strftime("%d-%b-%Y %H")

        # make flag from sunpy["logxmax1h"][i:i+24]
        
        if i+24 > len(sunpy["logxmax1h"]):
            # print(f"skip {data_time} because i+24 > len(sunpy['logxmax1h'])")
            continue
        logxmax1hs = sunpy["logxmax1h"].iloc[i:i+24]
        max_logxmax1h = max(logxmax1hs)
        # print(max_logxmax1h)

        # if max_logxmax1h is not a number, skip
        if np.isnan(max_logxmax1h):
            print(f"skip {date_time} because max_logxmax1h is np.nan")
            none_skip_count += 1
            continue

        flag = [0, 0, 0, 0]
        if max_logxmax1h < 0:
            flag[0] = 1
        elif max_logxmax1h >= 0 and max_logxmax1h < 1:
            flag[1] = 1
        elif max_logxmax1h >= 1 and max_logxmax1h < 2:
            flag[2] = 1
        elif max_logxmax1h >= 2:
            flag[3] = 1

        flag = ",".join([str(f) for f in flag])

        filename = sunpy["filename"].iloc[i]
            
        # filename : hmi_m_45s_2011_01_01_00_01_30_tai_magnetogram.fits.png
        year = int(filename.split("_")[3])
        month = int(filename.split("_")[4])
        
        new_record = {
            "time": date_time, 
            "aia131": None, 
            "aia1600": None,
            "magnetogram": f"../flare_transformer/data/noaa/magnetogram/{year}/{month:02d}/{filename}",
            "flag": flag,
            "feature": "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"
            # "feature": feature
        }

        data.append(new_record)

    # save data in jsonl format
    with open(output_path, "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")

    # print("None skip count:", none_skip_count)



def replace_sunpy_image2(dataset_path: str, csv_path: str, output_path: str):
    """
    Replace sunpy image
    """
    print("Load dataset ... ")
    data = read_jsonl(dataset_path)


    print("Load csv ... ")
    sunpy = pd.read_csv(csv_path)
    data_new = data.copy()

    remove_index = []

    for i, d in tqdm(enumerate(data), total=len(data)):
        data_time = data_new[i]["time"]
        data_time = datetime.datetime.strptime(data_time, "%d-%b-%Y %H")
        data_time = data_time.replace(minute=0, second=0, microsecond=0)

        if data_time < datetime.datetime.strptime("01-Jan-2011 00", "%d-%b-%Y %H"):
            # # remove i data
            # remove_index.append(i)
            continue
        elif data_time < datetime.datetime.strptime("01-Jan-2018 00", "%d-%b-%Y %H") and data_time >= datetime.datetime.strptime("01-Jan-2011 00", "%d-%b-%Y %H"):
            # print("skip", data_time)
            # replace sunpy data
            # get row index of sunpy["Time"] which is same as data_time
            sunpy_index = sunpy[sunpy["Time"] == data_time.strftime("%Y-%m-%d %H:%M:%S")].index
            if len(sunpy_index) == 0:
                remove_index.append(i)
                continue
                
            # filename : hmi_m_45s_2011_01_01_00_01_30_tai_magnetogram.fits.png
            year = int(sunpy["filename"].iloc[sunpy_index[0]].split("_")[3])
            month = int(sunpy["filename"].iloc[sunpy_index[0]].split("_")[4])
            data_new[i]["magnetogram"] = f"../flare_transformer/data/noaa/magnetogram/{year}/{month:02d}/{sunpy['filename'].iloc[sunpy_index[0]]}"

        else:
            # replace sunpy data
            # get row index of sunpy["Time"] which is same as data_time
            sunpy_index = sunpy[sunpy["Time"] == data_time.strftime("%Y-%m-%d %H:%M:%S")].index
            if len(sunpy_index) == 0:
                remove_index.append(i)
                continue

            # filename : hmi_m_45s_2011_01_01_00_01_30_tai_magnetogram.fits.png
            year = int(sunpy["filename"].iloc[sunpy_index[0]].split("_")[3])
            month = int(sunpy["filename"].iloc[sunpy_index[0]].split("_")[4])
            
            # data[i]["magnetogram"] = f"../flare_transformer/data/noaa/magnetogram/{year}/{month:02d}/{sunpy['filename'].iloc[sunpy_index[0]]}"
            data_new[i]["magnetogram"] = f"../flare_transformer/data/noaa/magnetogram/{year}/{month:02d}/{sunpy['filename'].iloc[sunpy_index[0]]}"
            # calculate flag
            flag = [0, 0, 0, 0]
            max_logxmax1h = max(sunpy["logxmax1h"].iloc[sunpy_index[0]:sunpy_index[0]+24])

            flag = ",".join([str(f) for f in flag])

            data_new[i]["flag"] = flag
            data_new[i]["feature"] = "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"
    # remove data
    data_new = [d for i, d in enumerate(data_new) if i not in remove_index]

    # save data in jsonl format using tqdm
    with open(output_path, "w") as f:
        for d in tqdm(data, total=len(data)):
            f.write(json.dumps(d) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="data/ft_database_all17.jsonl")
    parser.add_argument("--csv_path", type=str, default="data/noaa/magnetogram_logxmax1h_all_years.csv")
    parser.add_argument("--output_path", type=str, default="data/ft_database_all18_replace_logxmax1h.jsonl")
    args = parser.parse_args()

    replace_sunpy_image(**vars(args))


