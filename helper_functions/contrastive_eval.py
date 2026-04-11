import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve


def pair_scores_and_targets(model, batch):
    emb1 = model(**batch["enc1"])
    emb2 = model(**batch["enc2"])
    scores = F.cosine_similarity(emb1, emb2)
    targets = batch["same"]
    return scores.detach().cpu(), targets.detach().cpu()


# Source - https://stackoverflow.com/a/66549018
# Posted by Craig Bidstrup
# Retrieved 2026-04-09, License - CC BY-SA 4.0
def calibrate_threshold(y_true, scores):
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    numerator = 2 * recall * precision
    denom = recall + precision
    f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom!=0))
    return thresholds[np.argmax(f1_scores)]


def contrastive_metrics(score_list, target_list, threshold=None):
    scores = torch.cat(score_list).numpy()
    y_true = torch.cat(target_list).numpy()

    if threshold is None:
        threshold = calibrate_threshold(y_true, scores)

    preds = (scores >= threshold).astype(int)
    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds)
    return float(threshold), acc, f1
