import torch
import numpy as np
from sklearn.metrics import f1_score, roc_curve

def compute_accuracy(preds_float, labels, threshold=0.4084):
    valid_mask = labels != -1
    if valid_mask.sum() == 0: return 0.0
    preds_binary = (preds_float[valid_mask] >= threshold).float()
    correct = (preds_binary == labels[valid_mask]).sum().item()
    return correct / valid_mask.sum().item()

def compute_frame_eer(preds_float, labels):
    valid_mask = labels != -1
    if valid_mask.sum() == 0: return 0.0
    y_true = labels[valid_mask].cpu().numpy()
    y_scores = preds_float[valid_mask].cpu().numpy()
    # Need at least one of both classes to compute ROC
    if len(np.unique(y_true)) < 2: return 0.0
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.absolute((fnr - fpr)))
    return float(fpr[eer_idx])

def compute_spoof_jaccard_error_rate(preds_float, labels, threshold=0.4084):
    valid_mask = labels != -1
    if valid_mask.sum() == 0: return 0.0
    preds_binary = (preds_float[valid_mask] >= threshold).float()
    true_spoof = (labels[valid_mask] == 1.0)
    pred_spoof = (preds_binary == 1.0)
    intersection = (true_spoof & pred_spoof).sum().item()
    union = (true_spoof | pred_spoof).sum().item()
    if union == 0: return 0.0
    return 1.0 - (intersection / union)

def compute_macro_f1(preds_float, labels, threshold=0.4084):
    """
    Calculates Macro F1 to ensure the model isn't just predicting 'Spoof' 
    all the time to game the accuracy metric.
    """
    valid_mask = labels != -1
    if valid_mask.sum() == 0: return 0.0
    y_true = labels[valid_mask].cpu().numpy()
    y_pred = (preds_float[valid_mask].cpu().numpy() >= threshold).astype(int)
    if len(y_true) == 0: return 0.0
    return float(f1_score(y_true, y_pred, average='macro'))