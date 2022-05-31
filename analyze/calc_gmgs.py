import numpy as np
import json

with open("params/params_2017.json") as f:
    meta = json.load(f)

C = np.array([[7316, 141, 54, 14],
              [125, 78, 39, 43],
              [17, 34, 49, 45],
              [1, 0, 0, 35]])

S = np.array(meta["dataset"]["GMGS_score_matrix"])

N = C.sum()
f = "OCMX"
for i in range(4):
    for j in range(4):
        if i == j:
            continue
        gmgs = C[i, j] * (S[i, i] - S[i, j]) / N
        print(f"({f[i]},{f[j]}) -> {gmgs:.4f}")
