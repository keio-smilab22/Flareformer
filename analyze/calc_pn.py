import numpy as np

x = np.array([[7315, 139, 51, 20],
              [123, 76, 39, 47],
              [17, 34, 45, 49],
              [1, 0, 0, 35]])

s = "OCMX"
for i in range(4):
    tp = x[i, i]
    fp = x[:, i].sum() - tp
    fn = x[i, :].sum() - tp
    tn = x.sum() - (tp + fp + fn)
    print(f"{s[i]}: (tp,fp,fn,tn)=({tp},{fp},{fn},{tn}) -> {fp+fn}")
