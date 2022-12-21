import pandas as pd
import numpy as np
import json
import datetime


def calc_gmgs_score_matrix(path_sunpy_csv):
    """Calculate the GMGS score matrix from a csv file.

    Parameters
    ----------
    path_sunpy_csv : str
        Path to the csv file with the sunpy data.

    Returns
    -------
    list[list[float]]
        GMGS score matrix. 
    """

    df = pd.read_csv(path_sunpy_csv)
    # from df['Time'] = 2011-01-01 00:00:00 to 2016-12-31 23:00:00
    df_train = df[df['Time'] < '2018-01-01 00:00:00']

    # from df['Time'] = 2017-01-01 00:00:00 to 2018-12-31 23:00:00
    # df_train = df[df['Time'] >= '2017-01-01 00:00:00']


    # get logxmax1h
    logxmax1h = df_train['logxmax1h'].values

    labels = [] # 4 elements
    # the maximum value for 24 hours
    for i in range(0, len(logxmax1h)):
        max_value = max(logxmax1h[i:i+24])
        if max_value < 0:
            labels.append(0)
        elif max_value >= 0 and max_value < 1:
            labels.append(1)
        elif max_value >= 1 and max_value < 2:
            labels.append(2)
        else:
            labels.append(3)
    
    cnt = [0, 0, 0, 0]
    for i in range(4):
        cnt[i] = labels.count(i)
    
    p = []
    for i in range(4):
        p.append(cnt[i] / len(labels))
    print(f'p = {p}')

    gmgs_matrix = [[0 for i in range(4)] for j in range(4)]
    for i in range(1, 5):
        for j in range(i, 5):
            if i == j:
                gmgs_matrix[i - 1][i - 1] = s_ii(p, 4, i)
            else:
                gmgs_matrix[i - 1][j - 1] = gmgs_matrix[j - 1][i - 1] = s_ij(p, 4, i, j)
    return gmgs_matrix



def calc_gmgs_score_matrix_from_jsonl(path_jsonl):
    """Calculate the GMGS score matrix from a csv file.

    Parameters
    ----------
    path_jsonl : str
        Path to the jsonl file with the label data.

    Returns
    -------
    list[list[float]]
        GMGS score matrix. 
    """
    # open jsonl file
    with open(path_jsonl, 'r') as f:
        data = f.readlines()
    data = [json.loads(line) for line in data]

    # get labels(flag) to 

    labels = []
    for i in range(len(data)):
        # from start to 31-Nov-2016 23
        time = datetime.datetime.strptime(data[i]['time'], "%d-%b-%Y %H")
        if time < datetime.datetime.strptime('01-Jan-2016 00', "%d-%b-%Y %H"):
            labels.append(data[i]['flag'])
    # label
    # "1,0,0,0" -> 0
    # "0,1,0,0" -> 1
    # "0,0,1,0" -> 2
    # "0,0,0,1" -> 3
    cnt = {0: 0, 1: 0, 2: 0, 3: 0}
    for i in range(len(labels)):
        if labels[i] == '1,0,0,0':
            labels[i] = 0
            cnt[0] += 1
        elif labels[i] == '0,1,0,0':
            labels[i] = 1
            cnt[1] += 1
        elif labels[i] == '0,0,1,0':
            labels[i] = 2
            cnt[2] += 1
        elif labels[i] == '0,0,0,1':
            labels[i] = 3
            cnt[3] += 1
        else:
            print('ERROR')
            return 0
    p = []
    for i in range(4):
        p.append(cnt[i] / len(labels))
    print(f'p = {p}')
    print(f"p[0]+p[1]= {p[0]+p[1]}")
    print(f"p[2]+p[3]= {p[2]+p[3]}")

    gmgs_matrix = [[0 for i in range(4)] for j in range(4)]
    for i in range(1, 5):
        for j in range(i, 5):
            if i == j:
                gmgs_matrix[i - 1][i - 1] = s_ii(p, 4, i)
            else:
                gmgs_matrix[i - 1][j - 1] = gmgs_matrix[j - 1][i - 1] = s_ij(p, 4, i, j)
    return gmgs_matrix




def s_ii(p, N, i):
    sum_ak_inverse = 0
    sum_ak = 0
    for x in range(1, N):
        if x <= i - 1:
            sum_ak_inverse += 1 / a(p, x)
        else:
            sum_ak += a(p, x)
    return (sum_ak + sum_ak_inverse) / (N - 1)


def s_ij(p, N, i, j):
    if i > j:
        print("ERROR")
        return 0
    sum_ak_inverse = 0
    sum_ak = 0
    sum_minus = 0
    for x in range(1, N):
        if x <= i - 1:
            sum_ak_inverse += (1 / a(p, x))
        elif x <= j - 1:
            sum_minus += -1
        else:
            sum_ak += a(p, x)
    return (sum_ak + sum_ak_inverse + sum_minus) / (N - 1)


def a(p, i):
    sum_pk = 0
    for x in range(i):
        sum_pk += p[x]
    return (1-sum_pk) / sum_pk


if __name__ == '__main__':
    path_sunpy_csv = 'data/noaa/magnetogram_logxmax1h_all_years.csv'
    path_jsonl = 'data/ft_database_all17.jsonl'
    # gmgs_matrix = calc_gmgs_score_matrix(path_sunpy_csv)
    # print(gmgs_matrix)
    gmgs_matrix = calc_gmgs_score_matrix_from_jsonl(path_jsonl)
    print(gmgs_matrix)