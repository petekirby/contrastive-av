# pair_eval.py
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve


@torch.no_grad()
def get_pair_scores(model, dataloader, device=None):
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    y_true, scores = [], []
    for batch in dataloader:
        enc1 = {k: v.to(device) for k, v in batch["enc1"].items()}
        enc2 = {k: v.to(device) for k, v in batch["enc2"].items()}
        emb1 = model(**enc1)
        emb2 = model(**enc2)
        y_true.extend(batch["same"].tolist())
        scores.extend(F.cosine_similarity(emb1, emb2).cpu().tolist())
    return y_true, scores

# Source - https://stackoverflow.com/a/66549018
# Posted by Craig Bidstrup
# Retrieved 2026-04-09, License - CC BY-SA 4.0
def calibrate_threshold(y_true, scores):
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    numerator = 2 * recall * precision
    denom = recall + precision
    f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom!=0))
    return thresholds[np.argmax(f1_scores)]

def contrastive_evaluate(model, dataloader, threshold=None, device=None):
    y_true, scores = get_pair_scores(model, dataloader, device)
    if threshold is None:
        threshold = calibrate_threshold(y_true, scores)

    preds = [int(s >= threshold) for s in scores]
    return threshold, accuracy_score(y_true, preds), f1_score(y_true, preds)
