import torch
import numpy as np
from sklearn.metrics import roc_curve, f1_score, accuracy_score

def compute_eer(y_true, y_scores):
    """
    Calculates Equal Error Rate (EER) and the optimal decision threshold.
    y_true: array-like of shape (N,) containing 0.0 (bonafide) and 1.0 (spoof)
    y_scores: array-like of shape (N,) containing predicted probabilities or logits
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    if len(np.unique(y_true)) < 2:
        return 0.0, 0.5
        
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1.0 - tpr
    
    # Locate index where FPR and FNR are closest
    eer_idx = np.nanargmin(np.absolute(fnr - fpr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
    optimal_threshold = thresholds[eer_idx]
    
    return float(eer), float(optimal_threshold)


def compute_accuracy(y_true, y_scores, threshold=0.5):
    """
    Calculates classification accuracy given a decision threshold.
    """
    y_true = np.array(y_true)
    preds = (np.array(y_scores) >= threshold).astype(int)
    return float(accuracy_score(y_true, preds))


def compute_macro_f1(y_true, y_scores, threshold=0.5):
    """
    Calculates Macro F1-score given a decision threshold.
    """
    y_true = np.array(y_true)
    preds = (np.array(y_scores) >= threshold).astype(int)
    return float(f1_score(y_true, preds, average='macro'))c
