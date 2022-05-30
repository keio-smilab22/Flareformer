from asyncore import write
import csv
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

def stack_csv(path, save_path):
    path1 = path + '_train_feat.csv'
    path2 = path + '_validation_feat.csv'
    path3 = path + '_test_feat.csv'
    
    df1 = pd.read_csv(path1, header=None)
    df2 = pd.read_csv(path2, header=None)
    df3 = pd.read_csv(path3, header=None)

    df_new = pd.concat([df1, df2, df3], axis=0)
    df_new.to_csv(save_path, index=False)

    # with open(path1, 'r') as f1, open(path2, 'r') as f2, open(path3, 'r') as f3:
    #     reader1 = csv.reader(f1)
    #     print(len(list(reader1)))
    #     reader2 = csv.reader(f2)
    #     print(len(list(reader2)))
    #     reader3 = csv.reader(f3)
    #     print(len(list(reader3)))
        
    #     with open(save_path, 'w') as f:
    #         writer = csv.writer(f)
    #         for row1 in reader1:
    #             writer.writerows(row1)
    #         for row2 in reader2:
    #             writer.writerows(row2)
    #         for row3 in reader3:
    #             writer.writerows(row3)
    #         # save csv


def join_dataframe(path1, path2, save_path):
    df1 = pd.read_csv(path1, header=0)
    print(df1)
    df2 = pd.read_csv(path2) #stddev

    df1.rename(columns={'83': 'logXmax1h', '82': 'Xmax1h', '84':'logXmax1m1', '85':'logXmax1m2'}, inplace=True)
    print(df1.columns)
    df2.rename(columns={'logxmax1h':'logXmax1h', 'xmax1h': 'Xmax1h', 'logxmax1m1':'logXmax1m1', 'logxmax1m2':'logXmax1m2'}, inplace=True)
    tqdm.pandas()
    
    df_merged = pd.merge(df1, df2, on=['logXmax1h', 'Xmax1h', 'logXmax1m1', 'logXmax1m2'], how='right')
    df_merged.to_csv(save_path, index=False)

    # with open(path1, 'r') as f1, open(path2, 'r') as f2:
    #     reader1 = csv.reader(f1)
    #     reader2 = csv.reader(f2)
    #     l1 = [row for row in reader1]
    #     l2 = [row for row in reader2] #stddev

    #     with open(save_path, 'w') as f:
    #         start_idx = 1
    #         for i in range(1, len(l1)):
    #             for j in range(start_idx, len(l2)):
    #                 # print(i)
    #                 # print(type(l1[i][0]))
    #                 # str match
    #                 # print(f"{l1[i][83]} {l2[j][1]}")
    #                 if l1[i][83] == l2[j][1]:
    #                     l3 = l1[i] + l2[j]
    #                     # print(l3)
    #                     print(f"{i} {j} extend")
    #                     # save csv
    #                     writer = csv.writer(f)
    #                     writer.writerow(l3)
    #                     start_idx = j
    #                     break
                




if __name__ == '__main__':
    path = 'data/charval2017X_XMC24f'
    save_path = 'data/charval2017X_XMC24f_all.csv'
    
    # stack_csv(path, save_path)

    join_dataframe(path1=save_path, path2='data/stddev_charxxxall.csv', save_path='data/defn_feature_database_stddev.csv')

