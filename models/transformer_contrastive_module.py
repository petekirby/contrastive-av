# Significantly different model objectives/approaches will require their own modules on this pattern.
# The name of the module is one of the key YAML config items / command line arguments, used to select the model..
# Minor architecture swaps (e.g. different pretrained transformers) do not require separate modules.

import lightning.pytorch as pl
import torch
from transformers import get_scheduler
from .loss_defaults import build_loss_fn
from .optimizer_utils import build_param_groups
from .transformer_embedding_model import TransformerEmbeddingModel
from helper_functions.contrastive_eval import pair_scores_and_targets, contrastive_metrics
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score


# Source: https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
class TransformerContrastiveModule(pl.LightningModule):
    def __init__(
        self,
        model_config: dict,
        loss_dict: dict,
        lr: float = 2e-5,
        head_lr_multiplier: float = 5.0,
        weight_decay: float = 0.01,
        lr_schedule: str = "linear",
        warmup_ratio: float = 0.1,
        compile: bool = False,
    ):
        super().__init__()
        self.model = TransformerEmbeddingModel(**model_config)
        self.loss_fn, self.miner, loss_dict = build_loss_fn(loss_dict, embedding_size=self.model.embedding_dim)
        if compile:
            self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=False)
        self.register_buffer("eval_threshold", torch.tensor(float("nan")))
        self.test_acc = BinaryAccuracy()
        self.test_f1 = BinaryF1Score()
        self.pan20_test_acc = BinaryAccuracy()
        self.pan20_test_f1 = BinaryF1Score()

        self.save_hyperparameters(
            {
                "model_config": model_config,
                "loss_dict": loss_dict,
                "lr": lr,
                "head_lr_multiplier": head_lr_multiplier,
                "weight_decay": weight_decay,
                "lr_schedule": lr_schedule,
                "warmup_ratio": warmup_ratio,
            }
        )

    # format: {"input_ids": ..., "attention_mask": ..., sometimes "token_type_ids": ...}
    # usage: inputs = tokenizer(...), outputs = model(**inputs)
    def forward(self, **inputs) -> torch.Tensor:
        return self.model(**inputs)
    
    def loss(self, embeddings, target):
        if self.miner:
            mined = self.miner(embeddings, target)
            return self.loss_fn(embeddings, target, mined)
        else:
            return self.loss_fn(embeddings, target)

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        embeddings = self(**inputs)
        loss = self.loss(embeddings, target)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=target.shape[0])
        return loss

    def configure_optimizers(self):
        param_groups = build_param_groups(
            model=self.model,
            base_lr=self.hparams.lr,
            head_lr_multiplier=self.hparams.head_lr_multiplier,
            weight_decay=self.hparams.weight_decay,
            loss_fn=self.loss_fn,
            loss_optim_config=self.hparams.loss_dict["loss_optim_config"],
        )
        optimizer = torch.optim.AdamW(param_groups)
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = get_scheduler(
            name=self.hparams.lr_schedule,
            optimizer=optimizer,
            num_warmup_steps=int(total_steps * self.hparams.warmup_ratio),
            num_training_steps=total_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def on_validation_epoch_start(self):
        self._val_scores = []
        self._val_targets = []

    def validation_step(self, batch, batch_idx):
        scores, targets = pair_scores_and_targets(self, batch)
        self._val_scores.append(scores)
        self._val_targets.append(targets)

    def on_validation_epoch_end(self):
        threshold, acc, f1 = contrastive_metrics(self._val_scores, self._val_targets, threshold=None)
        self.eval_threshold.fill_(float(threshold))
        self.log("val_acc", acc, prog_bar=True, on_epoch=True)
        self.log("val_f1", f1, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        if torch.isnan(self.eval_threshold):
            raise RuntimeError("eval_threshold is not set")
        threshold = float(self.eval_threshold.item())
        scores, targets = pair_scores_and_targets(self, batch)
        preds = (scores >= threshold).int()
        targets = targets.int()
        if dataloader_idx == 0:
            self.test_acc.update(preds, targets)
            self.test_f1.update(preds, targets)
            self.log("test_acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
            self.log("test_f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        elif dataloader_idx == 1:
            self.pan20_test_acc.update(preds, targets)
            self.pan20_test_f1.update(preds, targets)
            self.log("pan20_test_acc", self.pan20_test_acc, on_step=False, on_epoch=True, prog_bar=False, add_dataloader_idx=False)
            self.log("pan20_test_f1", self.pan20_test_f1, on_step=False, on_epoch=True, prog_bar=False, add_dataloader_idx=False)
        else:
            raise ValueError(f"unexpected dataloader_idx: {dataloader_idx}")
