import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics

class GMGSRegressionLoss(nn.Module):
    def __init__(self):
        super(GMGSRegressionLoss, self).__init__()


    def forward(self, pred:torch.Tensor, true:torch.Tensor):
        """
        pred: (batch_size, 24 or 48, 1)
        true: (batch_size, 24 or 48, 1)
        """

        # convert to class
        pred_class = convert_to_class(pred)
        true_class = convert_to_class(true)

        # calculate confusion matrix
        gmgs_matrixs = []
        for i in range(true_class.shape[0]):

            confusion_matrix = metrics.confusion_matrix(true_class[i].detach().cpu().numpy(), pred_class[i].detach().cpu().numpy(), labels=[0, 1, 2, 3])

            gmgs_matrix = calc_gmgs_matrix(confusion_matrix, 4)
            gmgs_matrix = torch.Tensor(gmgs_matrix)
            
            gmgs_matrixs.append(gmgs_matrix)
        gmgs_matrixs = torch.stack(gmgs_matrixs, dim=0)
        # print(f"gmgs_matrixs: {gmgs_matrixs.shape}")

        # calculate gmgs loss
        gmgs_weight = torch.zeros_like(true_class, device=true.device)
        # print(f"gmgs_weight: {gmgs_weight.shape}")
        for i in range(true_class.shape[0]):
            for j in range(true_class.shape[1]):
                # print(f"true_class[i, j]: {true_class[i, j].item()}")
                gmgs_weight[i, j] = gmgs_matrixs[i, true_class[i, j].item(), pred_class[i, j].item()]

        # calculate weighted mse
        # print(f"squared error: {torch.square(pred - true).shape}")

        gmgs_loss = torch.sum(torch.square(pred - true) * gmgs_weight) / pred.shape[0]
        return gmgs_loss

def calc_gmgs_matrix(confusion_matrix, N):
    """
    Calculate GMGS matrix
    """
    p = [sum(x)/sum(map(sum, confusion_matrix)) for x in confusion_matrix]
    gmgs_matrix = [ [0 for i in range(N)] for j in range(N) ]
    for i in range(1, N+1):
        for j in range(i, N+1):
            if i == j:
                gmgs_matrix[i-1][i-1] = s_ii(p, N, i) 
            else:
                gmgs_matrix[i-1][j-1] = gmgs_matrix[j-1][i-1] = s_ij(p, N, i, j)
    # print(gmgs_matrix)
    return gmgs_matrix


        
def convert_to_class(pred:torch.Tensor) -> torch.Tensor:
    """
    Convert regression output to class
    """
    pred_class = torch.zeros((pred.shape[0], pred.shape[1]), dtype=torch.long)
    # print(pred_class.shape)
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            if pred[i, j, 0] >= 2:
                pred_class[i, j] = 3
            elif pred[i, j, 0] >= 1 and pred[i, j, 0] < 2:
                pred_class[i, j] = 2
            elif pred[i, j, 0] >= 0 and pred[i, j, 0] < 1:
                pred_class[i, j] = 1
            else:
                pred_class[i, j] = 0

    return pred_class

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

