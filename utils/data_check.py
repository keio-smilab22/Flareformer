import csv
import json
from unittest import result
import pandas as pd

def check_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in csv.reader(f):
            data.append(line[5])
    return data

def check_data(file_path, csv_path, num=0):
    """
    Check data from jsonl file and csv file
    file_path: path to jsonl file
    """

    # load jsonl file
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    # get data
    class_dict = {
        '1,0,0,0': 'O',
        '0,1,0,0': 'C',
        '0,0,1,0': 'M',
        '0,0,0,1': 'X',
    }

    df = pd.read_csv(csv_path, index_col='Time', parse_dates=True, date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'))
    
    df_charval = pd.read_csv('/home/katsuyuki/Downloads/charval2017X_XMC24f_train_feat.csv',  header=None)
    print(df_charval.head())

    # df['logXmax1h'] -> list
    # print(df['logXmax1h'])
    logxmax1h = df['logXmax1h'].tolist()

    logxmax1h_charval = df_charval[84].tolist()
    
    result = []
    for i in range(len(logxmax1h)):
        if round(logxmax1h[i], 4) != round(logxmax1h_charval[i], 4):
            result.append(i)
            if i == 0:
                print(logxmax1h[i], logxmax1h_charval[i])
    rate = len(result) / len(logxmax1h)
    print(rate)

    time = [d['time'] for d in data]
    # print(time[0:10])
    flag = [d['flag'] for d in data]
    flag = [class_dict[f] for f in flag]
    feature = [d['feature'] for d in data]
    
    # print(data[3])
    # print(len(feature[0].split(',')))
    # for i, f in enumerate(feature[0].split(',')):
    #     # print(f)

    # print(f"{num}:")
    # logxmax1h = [float(f.split(',')[85]) for f in feature]
    xmax1h = [float(f.split(',')[83]) for f in feature]
    # print(logxmax1h[:10])
    # print(xmax1h[:10])



    # feature classification
    logxmax1h_class = []
    

    # logXmax1h = log(Xmax1h) + 6
    for i, f in enumerate(logxmax1h):
        if i + 23 > len(logxmax1h):
            if i == len(logxmax1h):
                f_24 = logxmax1h[-1]
            else:
                f_24 = logxmax1h[i:len(logxmax1h)]
            
        else:
            f_24 = logxmax1h[i:i+23]
        # print(f_24)
        if max(f_24) >= 2:
            logxmax1h_class.append('X')
        elif max(f_24) >= 1 and max(f_24) < 2:
            logxmax1h_class.append('M')
        elif max(f_24) >= 0 and max(f_24) < 1:
            logxmax1h_class.append('C')
        elif max(f_24) < 0:
            logxmax1h_class.append('O')

    xmax1h_class = []
    for f in xmax1h:
        if f >= 1e-4:
            xmax1h_class.append('X')
        elif f >= 1e-5 and f < 1e-4:
            xmax1h_class.append('M')
        elif f >= 1e-6 and f < 1e-5:
            xmax1h_class.append('C')
        elif f < 1e-6:
            xmax1h_class.append('O')


    
    # difference between flag and logxmax1h_class
    diff = []
    _all = []
    for f, fc in zip(flag, logxmax1h_class):
        _all.append(f)
        if f != fc:
            diff.append(f+'-'+fc)
    print(f"ラベルの個数: {len(flag)}")
    print(f"featureの個数: {len(logxmax1h_class)}")
    print(f"計算結果と一致していない: {len(diff)}")
    o_c = diff.count('O-C')
    o_m = diff.count('O-M')
    o_x = diff.count('O-X')
    c_o = diff.count('C-O')
    c_m = diff.count('C-M')
    c_x = diff.count('C-X')
    m_o = diff.count('M-O')
    m_c = diff.count('M-C')
    m_x = diff.count('M-X')
    x_o = diff.count('X-O')
    x_c = diff.count('X-C')
    x_m = diff.count('X-M')

    print(f"ラベルはO，計算結果はC: {o_c}")
    print(f"ラベルはO，計算結果はM: {o_m}")
    print(f"ラベルはO，計算結果はX: {o_x}")
    print(f"ラベルはC，計算結果はO: {c_o}")
    print(f"ラベルはC，計算結果はM: {c_m}")
    print(f"ラベルはC，計算結果はX: {c_x}")
    print(f"ラベルはM，計算結果はO: {m_o}")
    print(f"ラベルはM，計算結果はC: {m_c}")
    print(f"ラベルはM，計算結果はX: {m_x}")
    print(f"ラベルはX，計算結果はO: {x_o}")
    print(f"ラベルはX，計算結果はC: {x_c}")
    print(f"ラベルはX，計算結果はM: {x_m}")

    print(f"total: {o_c+o_m+o_x+c_o+c_m+c_x+m_o+m_c+m_x+x_o+x_c+x_m}")


    # difference between flag and xmax1h_class
    diff_xmax1h = []
    for f, fc in zip(flag, xmax1h_class):
        if f != fc:
            diff_xmax1h.append(f+'-'+fc)
    # print(f"original: {len(flag)}")
    # print(f"違う: {len(diff_xmax1h)}")
    # print(diff_xmax1h)


def main():
    csv_path = '../data/data_all_v2.csv'
    file_path = '../data/ft_database_all17.jsonl'

    for i in range(1):
        check_data(file_path, csv_path, i)
    # check_data(file_path)


if __name__ == '__main__':
    main()