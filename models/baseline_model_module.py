# Responsibility: Henry

import pickle
import lightning.pytorch as pl
import numpy as np
import torch
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score
from helper_functions.contrastive_eval import calibrate_threshold

# TF-IDF and logistic regression
class BaselineModelModule(pl.LightningModule):
    def __init__(
        self,
        tfidf_config: dict,
        classifier_config: dict,
        pair_features: str = "abs_diff",
        negatives_per_positive: int = 1,
    ):
        super().__init__()
        if pair_features not in {"abs_diff", "abs_diff_product", "concat"}:
            raise ValueError(f"pair_features: {pair_features}")

        self.tfidf_config = dict(tfidf_config)
        if isinstance(self.tfidf_config.get("ngram_range"), list):
            self.tfidf_config["ngram_range"] = tuple(self.tfidf_config["ngram_range"])
        self.classifier_config = dict(classifier_config)
        self.pair_features = pair_features
        self.negatives_per_positive = negatives_per_positive

        self.vectorizer = TfidfVectorizer(**self.tfidf_config)
        self.model = LogisticRegression(**self.classifier_config)
        self.is_fitted = False

        # Threshold for converting predicted probability into binary decision
        self.register_buffer("eval_threshold", torch.tensor(0.5))
        self.automatic_optimization = False
        self.test_acc = BinaryAccuracy()
        self.test_f1 = BinaryF1Score()
        self.pan20_test_acc = BinaryAccuracy()
        self.pan20_test_f1 = BinaryF1Score()

        self.save_hyperparameters(
            {
                "tfidf_config": self.tfidf_config,
                "classifier_config": self.classifier_config,
                "pair_features": pair_features,
                "negatives_per_positive": negatives_per_positive,
            }
        )

    # Convert two raw text lists into pair-level sparse features
    def pair_matrix(self, texts1, texts2):
        x1 = self.vectorizer.transform(texts1)
        x2 = self.vectorizer.transform(texts2)

        if self.pair_features == "concat":
            return sparse.hstack([x1, x2], format="csr")

        abs_diff = abs(x1 - x2)
        if self.pair_features == "abs_diff_product":
            return sparse.hstack([abs_diff, x1.multiply(x2)], format="csr")
        return abs_diff

    # Fit TF-IDF vocabulary and logistic regression on sampled training pairs
    def fit(self, texts1, texts2, labels):
        self.vectorizer.fit(texts1 + texts2)
        features = self.pair_matrix(texts1, texts2)
        self.model.fit(features, labels)
        self.is_fitted = True

    # Return same author probabilities from the fitted model
    def predict_scores(self, texts1, texts2):
        if not self.is_fitted:
            raise RuntimeError("Baseline model has not been fitted yet")
        features = self.pair_matrix(texts1, texts2)
        return self.model.predict_proba(features)[:, 1]

    # Inputs are raw text instead of token IDs
    def forward(self, text1, text2) -> torch.Tensor:
        scores = self.predict_scores(text1, text2)
        return torch.tensor(scores, dtype=torch.float32, device=self.device)

    def configure_optimizers(self):
        return None

    # Collect all training pairs for epoch-level sklearn fitting
    def on_train_epoch_start(self):
        print("Starting training epoch...")
        self._train_texts1 = []
        self._train_texts2 = []
        self._train_targets = []

    # Store raw text pairs from each batch, fitting happens once at epoch end
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        self._train_texts1.extend(inputs["text1"])
        self._train_texts2.extend(inputs["text2"])
        self._train_targets.append(labels.detach().cpu().numpy().astype(int))

    # Fit the vectorizer and classifier after collecting the epoch's pairs
    def on_train_epoch_end(self):

        print("\nStarting end of epoch training...")

        labels = np.concatenate(self._train_targets)
        print(f"Number of training pairs: {len(self._train_texts1)}")

        print("Fitting TF-IDF vectorizer...")
        self.vectorizer.fit(self._train_texts1 + self._train_texts2)

        print("Transforming text pairs into features...")
        features = self.pair_matrix(self._train_texts1, self._train_texts2)
        print(f"Feature matrix shape: {features.shape}")

        print("Training logistic regression...")
        self.model.fit(features, labels)

        # Param count
        num_logreg_params = self.model.coef_.size + self.model.intercept_.size
        print(f"Number of sklearn parameters: {num_logreg_params}")

        print("Training complete!")
        self.is_fitted = True

        del self._train_texts1, self._train_texts2, self._train_targets

    # Collect validation scores for threshold calibration
    def on_validation_epoch_start(self):
        self._val_scores = []
        self._val_targets = []
        self._train_metric_scores = []
        self._train_metric_targets = []

    # Score validation pairs
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if not self.is_fitted:
            return
        inputs, labels = batch
        scores = self.predict_scores(inputs["text1"], inputs["text2"])
        labels = labels.detach().cpu().numpy().astype(int)

        if dataloader_idx == 0:
            self._val_scores.append(scores)
            self._val_targets.append(labels)
        elif dataloader_idx == 1:
            self._train_metric_scores.append(scores)
            self._train_metric_targets.append(labels)
        else:
            raise ValueError(f"unexpected dataloader_idx: {dataloader_idx}")
        
        if batch_idx % 500 == 0:
            print(f"Validation batch {batch_idx}")

    # Calibrate threshold on validation F1, then log validation/train metrics
    def on_validation_epoch_end(self):
        if not self.is_fitted:
            return
        val_scores = np.concatenate(self._val_scores)
        val_targets = np.concatenate(self._val_targets)
        threshold = calibrate_threshold(val_targets, val_scores)
        self.eval_threshold.fill_(float(threshold))

        val_preds = (val_scores >= threshold).astype(int)
        self.log("val_acc", accuracy_score(val_targets, val_preds), prog_bar=True, on_epoch=True, add_dataloader_idx=False)
        self.log("val_f1", f1_score(val_targets, val_preds), prog_bar=True, on_epoch=True, add_dataloader_idx=False)

        train_scores = np.concatenate(self._train_metric_scores)
        train_targets = np.concatenate(self._train_metric_targets)
        train_preds = (train_scores >= threshold).astype(int)
        self.log("train_acc", accuracy_score(train_targets, train_preds), prog_bar=False, on_epoch=True, add_dataloader_idx=False)
        self.log("train_f1", f1_score(train_targets, train_preds), prog_bar=False, on_epoch=True, add_dataloader_idx=False)

    # Two test dataloaders: 0 = PAN21 (primary test set), 1 = PAN20
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        inputs, labels = batch
        scores = self.predict_scores(inputs["text1"], inputs["text2"])
        labels = labels.int()
        preds = torch.tensor((scores >= float(self.eval_threshold.item())).astype(int), dtype=torch.int, device=labels.device)

        if dataloader_idx == 0:
            self.test_acc.update(preds, labels)
            self.test_f1.update(preds, labels)
            self.log("test_acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
            self.log("test_f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        elif dataloader_idx == 1:
            self.pan20_test_acc.update(preds, labels)
            self.pan20_test_f1.update(preds, labels)
            self.log("pan20_test_acc", self.pan20_test_acc, on_step=False, on_epoch=True, prog_bar=False, add_dataloader_idx=False)
            self.log("pan20_test_f1", self.pan20_test_f1, on_step=False, on_epoch=True, prog_bar=False, add_dataloader_idx=False)
        else:
            raise ValueError(f"unexpected dataloader_idx: {dataloader_idx}")

    # Store sklearn state inside checkpoints
    def on_save_checkpoint(self, checkpoint):
        checkpoint["baseline_model_state"] = pickle.dumps(
            {
                "vectorizer": self.vectorizer,
                "model": self.model,
                "is_fitted": self.is_fitted,
            }
        )

    # Restore fitted sklearn objects from checkpoints
    def on_load_checkpoint(self, checkpoint):
        state_bytes = checkpoint.get("baseline_model_state")
        if state_bytes is None:
            return
        state = pickle.loads(state_bytes)
        self.vectorizer = state["vectorizer"]
        self.model = state["model"]
        self.is_fitted = state["is_fitted"]
