"""Script for evaluation metrics"""

import math
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from numpy import ndarray
from sklearn import metrics


class Stat:
    """
    collect predictions and calculate some metrics
    """

    def __init__(self, climatology: List[float]) -> None:
        self.predictions = []
        self.observations = []
        self.climate = climatology

    def collect(self, pred: torch.Tensor, ground_truth: torch.Tensor):
        """
        Collect predictions and ground truth
        """
        observation = torch.argmax(ground_truth, dim=1)
        self.predictions.extend(pred.cpu().detach().numpy())
        self.observations.extend(observation.cpu().detach().numpy())

    def aggregate(self, dataset_type: str) -> Dict[str, Any]:
        """
        Aggregate collected data and calculate metrics
        """
        score = {}
        y_pred, y_true = np.array(self.predictions), np.array(self.observations)
        y_predl = [np.argmax(y) for y in y_pred]

        score["ACC"] = self.calc_accs(y_predl, y_true)
        score["TSS-M"] = self.calc_tss(y_predl, y_true, 2)
        score["BSS-M"] = self.calc_bss(y_pred, y_true, self.climate)
        score["GMGS"] = self.calc_gmgs(y_predl, y_true)

        results = {dataset_type + "_" + k: v for k, v in score.items()}

        return results

    def clear_all(self):
        """
        Clear all collected data
        """
        self.predictions.clear()
        self.observations.clear()

    def calc_mattheus(self, y_predl: ndarray, y_true: ndarray, flare_class: int):
        """
        Compute Matthews correlation coefficient
        """
        C = self.confusion_matrix(y_predl, y_true)
        c, s = np.diag(C).sum(), C.sum()
        p, t = np.sum(C, axis=0), np.sum(C, axis=1)
        mcc = (c * s - np.dot(p, t).sum()) / np.sqrt((s**2 - np.dot(p, p).sum()) - (s**2 - np.dot(t, t).sum()))
        return mcc

    def calc_tss(self, y_predl: ndarray, y_true: ndarray, flare_class: int) -> float:
        """
        Compute TSS
        """
        mtx = self.confusion_matrix(y_predl, y_true)
        tn, fp, fn, tp = self.binary_confusion_matrix(mtx, flare_class)
        tss = (tp / (tp + fn)) - (fp / (fp + tn))
        return float(tss) if not math.isnan(tss) else 0.0

    def calc_gmgs(self, y_predl: ndarray, y_true: ndarray) -> float:
        """
        Compute GMGS (simplified version)
        """
        s = "XMCO"
        tss = np.empty(3, dtype=np.float)
        for i in range(3):
            tss[i] = self.calc_tss(y_predl, y_true, 3 - i)
            print(f"{s[i]}: {tss[i]}")

        return tss.mean()

    def calc_bss(self, y_pred: ndarray, y_true: ndarray, climatology: List[float]) -> float:
        """
        Compute BSS >= M
        """
        N = len(y_true)
        y_truel = np.array([self.convert_binary_onehot(y) for y in y_true])
        y_pred = np.vstack([y_pred[:, :2].sum(axis=1), y_pred[:, 2:].sum(axis=1)]).transpose()

        d = y_pred[:, 1] - y_truel[:, 1]
        bs = np.dot(d, d) / N
        bsc = climatology[0] * climatology[1]
        bss = (bsc - bs) / bsc

        return bss

    def convert_binary_onehot(self, flare_class: int) -> List[int]:
        """
        return 2-dimentional 1-of-K vector
        """
        return np.array([1, 0] if flare_class < 2 else [0, 1])

    def confusion_matrix(self, y_pred: ndarray, y_true: ndarray) -> ndarray:
        """
        return confusion matrix for 4 class
        """
        return metrics.confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])

    def calc_accs(self, y_pred: ndarray, y_true: ndarray) -> float:
        """
        Compute classification accuracy for 4 class
        """
        cm = self.confusion_matrix(y_pred, y_true)
        accs = np.diag(cm).sum() / len(y_pred)
        print(cm)
        return accs

    def binary_confusion_matrix(self, mtx: ndarray, target_class: int) -> Tuple[int, int, int, int]:
        """
        convert confusion matrix for 2 class ("< target_class" or ">= target_class")
        """

        tn, fp = mtx[:target_class, :target_class].sum(), mtx[:target_class, target_class:].sum()
        fn, tp = mtx[target_class:, :target_class].sum(), mtx[target_class:, target_class:].sum()

        return tn, fp, fn, tp
