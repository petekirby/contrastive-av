# Responsibility: Eric

# This should be a lightning module built on the pattern of transformer_contrastive_module.py
# You can use a model or model+head from transformer_embedding_model.py for most of the model part.
# Then this file could add a linear classification head on top of it (and handle classification instead of contrastive learning objectives).

import lightning.pytorch as pl
import torch
import torch.nn as nn
from transformers import get_scheduler
from .transformer_embedding_model import TransformerEmbeddingModel
from .optimizer_utils import split_weight_decay_params
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score

class TransformerClassificationModule(pl.LightningModule):
    def __init__(
        self,
        model_config: dict,
        lr: float = 2e-5,
        head_lr_multiplier: float = 5.0,
        weight_decay: float = 0.01,
        lr_schedule: str = "linear",
        warmup_ratio: float = 0.1,
        compile: bool = False,
        freeze_encoder: bool = False,
    ):
        super().__init__()

        # Reuse the same embedding model as contrastive module (encoder + pooling + projection)
        self.model = TransformerEmbeddingModel(**model_config)

        # Classification head, responsible for mapping embedding to 2 classes (same author / different author)
        self.classifier = nn.Linear(self.model.embedding_dim, 2)

        # Standard CE loss for classification purposes
        self.loss_fn = nn.CrossEntropyLoss()

        # Enable transformer encoder freezing (only head trains)
        if freeze_encoder:
            for param in self.model.encoder.parameters():
                param.requires_grad = False 

        # Also enable speed / efficiency improvements via torch.compile
        if compile:
            self.model = torch.compile(self.model, mode = "reduce-overhead", fullgraph = False)

        # Metrics, separate instances for val/PAN20/21 test
        self.val_acc = BinaryAccuracy()
        self.val_f1 = BinaryF1Score()
        self.test_acc = BinaryAccuracy()
        self.test_f1 = BinaryF1Score()
        self.pan20_test_acc = BinaryAccuracy()
        self.pan20_test_f1 = BinaryF1Score()        

        # Save hyperparameters for logging purposes 
        self.save_hyperparameters({
            "model_config": model_config,
            "lr": lr,
            "head_lr_multiplier": head_lr_multiplier,
            "weight_decay": weight_decay,
            "lr_schedule": lr_schedule,
            "warmup_ratio": warmup_ratio,
            "freeze_encoder": freeze_encoder,
        })

    # Get pooled embedding from transformer, then classify
    def forward(self, **inputs) -> torch.Tensor:
        embeddings = self.model(**inputs)
        logits = self.classifier(embeddings)
        return logits

    # Unpack tokenized pair dict and binary labels
    def training_step(self, batch, batch_idx): # batch_idx required by Lightning module even if unused
        inputs, labels = batch
        logits = self(**inputs)
        loss = self.loss_fn(logits, labels)
        self.log("train_loss", loss, prog_bar = True, on_step = False, on_epoch = True, batch_size = labels.shape[0])
        return loss    
    
    # Compute predictions and accumulate metrics
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self(**inputs)
        preds = logits.argmax(dim = -1)
        self.val_acc.update(preds, labels)
        self.val_f1.update(preds, labels)

        # Also log validation loss for monitoring
        loss = self.loss_fn(logits, labels)
        self.log("val_loss", loss, prog_bar = False, on_step = False, on_epoch = True, batch_size = labels.shape[0])

    # Compute and log accumulated val metrics, then reset for next epoch
    def on_validation_epoch_end(self):
        self.log("val_acc", self.val_acc.compute(), prog_bar=True)
        self.log("val_f1", self.val_f1.compute(), prog_bar=True)
        self.val_acc.reset()
        self.val_f1.reset()
    
    # Two test dataloaders: 0 = PAN21 (primary test set), 1 = PAN20
    def test_step(self, batch, batch_idx, dataloader_idx = 0):
        inputs, labels = batch
        logits = self(**inputs)
        preds = logits.argmax(dim=-1)

        if dataloader_idx == 0:
            # PAN21 test set (open-set: unseen authors and topics)
            self.test_acc.update(preds, labels)
            self.test_f1.update(preds, labels)
            self.log("test_acc", self.test_acc, on_step = False, on_epoch = True, prog_bar = True, add_dataloader_idx = False)
            self.log("test_f1", self.test_f1, on_step = False, on_epoch = True, prog_bar = True, add_dataloader_idx = False)
        elif dataloader_idx == 1:
            # PAN20 test set (closed-set: authors/topics seen during training)
            self.pan20_test_acc.update(preds, labels)
            self.pan20_test_f1.update(preds, labels)
            self.log("pan20_test_acc", self.pan20_test_acc, on_step = False, on_epoch = True, prog_bar = False, add_dataloader_idx = False)
            self.log("pan20_test_f1", self.pan20_test_f1, on_step = False, on_epoch = True, prog_bar = False, add_dataloader_idx = False)
        else:
            raise ValueError(f"unexpected dataloader_idx: {dataloader_idx}")       


    # Build param groups with different learning rates
    # Encoder params: base lr (small as a means of preserving pre-trained knowledge)
    # Projection + classifier params: lr * multiplier (larger since it's learning from scratch)
    def configure_optimizers(self):
        param_groups = []

        # Encoder parameters (will be empty if frozen since requires_grad = False)
        param_groups.extend(
            split_weight_decay_params(
                self.model.encoder.named_parameters(),
                lr = self.hparams.lr,
                weight_decay = self.hparams.weight_decay,
            )
        )

        # Projection head parameters (from TransformerEmbeddingModel)
        head_lr = self.hparams.lr * self.hparams.head_lr_multiplier
        param_groups.extend(
            split_weight_decay_params(
                self.model.projection.named_parameters(),
                lr = head_lr,
                weight_decay = self.hparams.weight_decay,
            )
        )

        # Classification head parameters
        param_groups.extend(
            split_weight_decay_params(
                self.classifier.named_parameters(),
                lr = head_lr,
                weight_decay=self.hparams.weight_decay,
            )
        )

        # Filter out empty groups (i.e. encoder parameters when freezing)
        param_groups = [g for g in param_groups if g["params"]]

        # AdamW optimizer with linear warmup + decay schedule
        optimizer = torch.optim.AdamW(param_groups)
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = get_scheduler(
            name = self.hparams.lr_schedule,
            optimizer = optimizer,
            num_warmup_steps = int(total_steps * self.hparams.warmup_ratio),
            num_training_steps = total_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
