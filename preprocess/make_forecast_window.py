import csv
import json


def make_window(
        year: int, 
        compressedDayList: list,) -> list:
    """
        2011年と2017年の最初は0を重複させる
        その他はflareformerの実装上負の値で補間
        >> というのも2011年からのindex Nを年ごとに足しているため
    """
    windows = []
    num = 0

    YearList = [2011, 2017]

    for _, day in enumerate(compressedDayList):
        window = [0] * 24
        # 24時間分ある場合にはそのまま追加
        if int(day) == 24:
            for idx in range(day):
                window[idx] = num
                num += 1
        
        # 24時間分ない場合
        else:
            dif = 24 - day
            # 2011年 or 2017年の場合
            if (num == 0 and (year in YearList)):
                for idx in range(dif, 24):
                    window[idx] = num
                    num += 1
            else:
                num -= dif
                for idx in range(24):
                    window[idx] = num
                    num += 1
        
        windows.append(sorted(window, reverse=True))
    
    return windows


def main():
    # flareformerの情報をもつjsonファイルを読み込む
    json_path = "/home/initial/Flareformer/data_forecast/ft_database_all17.jsonl"
    with open(json_path, 'r') as f:
        lines = f.readlines()

    days = []
    for idx, line in enumerate(lines):
        line = json.loads(line)
        # mag: ../flare_transformer/data/magnetogram/2010-06/hmi.M_720s.20100601_015825.magnetogram.png
        date = line['magnetogram'].split('/')[-1].split('.')[-3].split('_')[0]
        day = int(date[-2:])
        
        # 2010年はskip
        if date[:4] == "2010":
            continue
        # if date[:4] == "2012": break
        days.append(day)
    
    '''
    days: 
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
     3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
     4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 
     5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 
     6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 
     7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 
     8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 
     9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 
     10, 10, ...]
    '''
    
    # 連長圧縮
    days_comp = {}
    for year in range(2011, 2018):
        days_comp[year] = []
    
    year = 2011
    count = 1
    i = 0
    month = 0
    
    while (i<len(days)):
        # year の更新
        if (month==12 and int(days[i])==1 and int(days[i-1])!=1):
            year += 1
            month = 1
        
        # monthの更新
        elif (month!=12 and int(days[i])==1 and int(days[i-1])!=1):
            month += 1
        
        # 連長圧縮
        # flareformerの実装上2011と2017以外は負の数があってもよさそう
        # というのもtrainでは2011からの数Nを足すため
        j = i+1
        while (j<len(days) and days[i]==days[j]):
            count += 1
            j += 1
        days_comp[year].append(count)
        
        # init
        i = j
        count = 1
    
    print('days >>> ', len(days))
    print('days_compression >>>')
    for year in range(2011, 2018):
        print(f"{year} size :", len(days_comp[year]))
    days = days_comp

    # 同じ日ごとのwindowの作成
    # 24時間ごとで分割
    print('make window >>>')
    
    for year in days:
        day_by_year = days[year]
        # TODO: 多分indexが1大きい
        windows = make_window(year, day_by_year)
        print(f'window size {year}', len(windows))
    
        save_path = f"/home/initial/Flareformer/data_forecast/data_{year}_window_forecast.csv"
        with open(save_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerows(windows)
    
    print('===== Finish window to csv ======')



if __name__ == "__main__":
    main()