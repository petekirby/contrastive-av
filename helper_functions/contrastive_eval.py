# It's not clear that classification will need much similar eval code, so classification_eval.py may not be necessary.

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat


def pair_scores_and_targets(model, batch):
    emb1 = model(**batch["enc1"]).clone()
    emb2 = model(**batch["enc2"]).clone()
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
    scores = torch.cat(score_list).detach().cpu().numpy()
    y_true = torch.cat(target_list).detach().cpu().numpy()

    if threshold is None:
        threshold = calibrate_threshold(y_true, scores)

    preds = (scores >= threshold).astype(int)
    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds)
    return float(threshold), acc, f1


# Source: https://lightning.ai/docs/torchmetrics/stable/pages/implement.html
class ContrastiveEvalMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("scores", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, scores: torch.Tensor, targets: torch.Tensor):
        self.scores.append(scores.detach().flatten())
        self.targets.append(targets.detach().flatten())

    def compute(self, threshold=None):
        scores = dim_zero_cat(self.scores)
        targets = dim_zero_cat(self.targets)
        return contrastive_metrics([scores], [targets], threshold=threshold)
