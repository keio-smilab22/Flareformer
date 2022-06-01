import matplotlib.pyplot as plt
import numpy as np
import json

def read_meta_jsonl():
    with open("images/ft_database_all17.jsonl", "r") as f:
        return [json.loads(line) for line in f]

meta = read_meta_jsonl()

timestampss = [["21-Jan-2017 00","21-Jan-2017 03"], ["01-Apr-2017 14","01-Apr-2017 17"], ["05-Sep-2017 06","05-Sep-2017 09"]]
label = ["MC","MM","XX"]
for i, timestamps in enumerate(timestampss):
    for timestamp in timestamps:
        for line in meta:
            if timestamp in line["time"]:
                feature = list(map(float,line["feature"].split(",")))
                N = len(feature)
                x = np.linspace(0,N-1,N)
                plt.bar(x,feature,width=2.5, color="#000000")
                plt.savefig(f"images/bar_{label[i]}{timestamp}.png")