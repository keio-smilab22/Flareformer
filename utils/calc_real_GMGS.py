import sys
import sklearn.metrics as metrics
import numpy as np
from pprint import pprint

# cm = \
# [[1458,  348,  160,  325],
#  [ 723,  736,  845, 2180],
#  [  47,   77,  158,  921],
#  [   0,    9,   11,  128]]
# ^ id13-c, 
# x:  0.43543392979341844
# m:  -0.015418865326452286
# c:  0.04497386766836116
# 0.15499631071177575

# cm = \
# [[1363,  326,  321,  281],
#  [ 428, 1097, 1134, 1825],
#  [  22,  155,  277,  749],
#  [   0,   11,   25,  112]]
# ^ id5
# x:  0.39889764419721807
# m:  0.01647753599774457
# c:  0.10955702648406121
# 0.17497740222634128


# cm = \
# [[1236, 1032,   23,    0],
#  [ 483, 3620,  318,   63],
#  [  20,  789,  264,  130],
#  [   1,   46,   49,   52]]
# ^ id0-c
# x:  0.3271598246529307
# m:  0.16311741230485824
# c:  0.29468447456843316
# 0.26165390384207404

cm = \
[[1979, 419,   21,    2],
 [ 335, 1554,  453,   11],
 [  23,  379,  495,  43],
 [   0,   19,   82,   29]]


# cm = \
# [[6707,  460,   48,    0],
#  [  36,  199,   50,    0],
#  [   6,   63,   75,    1],
#  [   1,    0,   33,    2]]
# ^ id0, DeFN all17

# cm = \
# [[6772,  314,  129,    0],
#  [  32,  175,   78,    0],
#  [   5,   44,   96,    0],
#  [   0,    1,   33,    2]]
# ^ id4-b


# cm = \
# [[6731,  444,   40,    0],
#  [  78,  193,   14,    0],
#  [  13,   55,   77,    0],
#  [   1,    0,   33,    2]]
# ^ id7

# cm = \
# [[7017,  195,    0,    3],
#  [ 112,  172,    1,    0],
#  [  14,   98,   17,   16],
#  [   0,    0,    0,   36]]

# cm = \
# [[6400,  557,  179,   79],
#  [  48,  106,   67,   64],
#  [   0,   28,   84,   33],
#  [   0,    0,    0,   36]]
# ^ id19


# cm = \
# [[6412,  578,    0],
#  [ 677,   37,    0],
#  [  91,    0,    0]]

# cm = \
# cm = \
# cm = \
# cm = \
# cm = \


def calc_tp_4(four_class_matrix, flare_class):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for true in range(4):
        for pred in range(4):
            if true >= flare_class:
                if pred >= flare_class:
                    tp += four_class_matrix[true][pred]
                else:
                    fn += four_class_matrix[true][pred]
            else:
                if pred >= flare_class:
                    fp += four_class_matrix[true][pred]
                else:
                    tn += four_class_matrix[true][pred]
    return tn, fp, fn, tp

def calc_tp_1(four_class_matrix, flare_class):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for true in range(4):
        for pred in range(4):
            if true == flare_class:
                if pred == flare_class:
                    tp += four_class_matrix[true][pred]
                else:
                    fn += four_class_matrix[true][pred]
            else:
                if pred == flare_class:
                    fp += four_class_matrix[true][pred]
                else:
                    tn += four_class_matrix[true][pred]
    return tn, fp, fn, tp

def calc_TSS4(cm, flare_class):
    tn, fp, fn, tp = calc_tp_4(cm, flare_class)
    tss = (tp / (tp+fn)) - (fp / (fp + tn))
    return float(tss)

def calc_GMGS(cm):
    tss_x = calc_TSS4(cm, 3)
    tss_m = calc_TSS4(cm, 2)
    tss_c = calc_TSS4(cm, 1)
    print("x: ", tss_x)
    print("m: ", tss_m)
    print("c: ", tss_c)
    return (tss_x + tss_m + tss_c) / 3


# cm = \
# [[1979, 419,   21,    2],
#  [ 335, 1554,  453,   11],
#  [  23,  379,  495,  43],
#  [   0,   19,   82,   29]]
def a(p, i):
    sum_pk = 0
    for x in range(i):
        sum_pk += p[x]
    return (1-sum_pk) / sum_pk

def s_ii(p, N, i):
    sum_ak_inverse = 0
    sum_ak = 0
    for x in range(1, N):
        if x <= i - 1:
            sum_ak_inverse += 1 / a(p, x)
        else:
            sum_ak += a(p, x)
    return (sum_ak + sum_ak_inverse) / (N-1)

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
    return (sum_ak + sum_ak_inverse + sum_minus) / (N-1)

def calc_gmgs_matrix(cm, N):
    p = [sum(x)/sum(map(sum, cm)) for x in cm]
    # print(sum(map(sum, cm)))
    print(f'p: {p}')
    gmgs_matrix = [ [0 for i in range(N)] for j in range(N) ]
    for i in range(1, N+1):
        for j in range(i, N+1):
            if i == j:
                gmgs_matrix[i-1][i-1] = s_ii(p, N, i) 
            else:
                gmgs_matrix[i-1][j-1] = gmgs_matrix[j-1][i-1] = s_ij(p, N, i, j)
    print(f'{gmgs_matrix[0]}\n{gmgs_matrix[1]}\n{gmgs_matrix[2]}\n{gmgs_matrix[3]}')
    return gmgs_matrix

if __name__ == "__main__":
    print(calc_gmgs_matrix(cm, 4))
    print(calc_GMGS(cm))
    sys.exit()

    p = [sum(x)/sum(map(sum, cm)) for x in cm]
    print(p)
    # aa = a(p, 4)
    # print(aa)
    # l = []
    # for i in range(1, 5):
    #     ss = s_ii(p, 4, i)
    #     l.append(ss)
    # print(l)
    # ll = []
    # for i in range(4):
    #     ll.append(l[i] * p[i])
    # print("ll :", ll, sum(ll))

    score_matrix = np.array(calc_gmgs_matrix(cm, 4))
    print(calc_gmgs_matrix(cm, 4))
    cm_prob = np.array(cm) / sum(map(sum, cm))
    # print(cm_prob)
    print(score_matrix)
    # sys.exit()
    gmgs = np.trace(np.matmul(score_matrix, cm_prob))
    print(gmgs)

    # tmp = score_matrix[1]
    # tmp_sum = 0
    # for i in range(4):
    #     tmp_sum += tmp[i] * p[i]
    # print("tmp_sum  ", tmp_sum)

    # print("s_23  ", s_ij(p, 4, 2, 3))
    # print("a1_-1  ", 1/a(p, 1))
    # print("a3  ", a(p, 3))
    # x = 1/a(p, 1) + a(p, 3) - 2
    # print(x)

    # print(calc_GMGS(cm))



