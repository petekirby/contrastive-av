# Responsibility: Eric

import lightning.pytorch as pl
import torch
import bitsandbytes as bnb
import torch.nn as nn
from kornia.losses import BinaryFocalLossWithLogits
from transformers import AutoModelForSequenceClassification, get_scheduler
from .optimizer_utils import split_weight_decay_params
from helper_functions.contrastive_eval import calibrate_threshold
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score

class TransformerClassificationModule(pl.LightningModule):
    def __init__(
        self,
        model_config: dict,
        lr: float = 2e-5,
        head_lr_multiplier: float = 5.0,
        weight_decay: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        lr_schedule: str = "linear",
        warmup_ratio: float = 0.1,
        compile: bool = False,
        freeze_encoder: bool = False,
        negatives_per_positive: int = 1,
        gamma: float = 0.0,
    ):
        super().__init__()

        # Use HF's sequence classification model, which preserves model-specific intermediate layers 
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_config["model_name_or_path"],
            num_labels = 1, # Single logit for binary classification with BCE loss
            attn_implementation = model_config.get("attn_implementation"),
            dtype = torch.bfloat16 if model_config.get("use_bf16") else None
        )

        # Focal loss on a single logit 
        alpha = negatives_per_positive / (negatives_per_positive + 1) # class balanced
        self.loss_fn = BinaryFocalLossWithLogits(alpha=alpha, gamma=gamma, reduction="mean")

        # Enable transformer encoder freezing 
        if freeze_encoder:
            for name, param in self.model.named_parameters():
                # Keep classifier head trainable, freeze everything else
                if "classifier" not in name:
                    param.requires_grad = False 

        # Also enable speed / efficiency improvements via torch.compile
        if compile:
            self.model = torch.compile(self.model, mode = "default", fullgraph = False)

        # Threshold for converting logit to prediction, calibrated on validation set                                                                         
        self.register_buffer("eval_threshold", torch.tensor(0.0))

        # Optimizer hyperparameters
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

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
            "beta1": beta1,
            "beta2": beta2,
            "eps": eps,
            "lr_schedule": lr_schedule,
            "warmup_ratio": warmup_ratio,
            "freeze_encoder": freeze_encoder,
            "negatives_per_positive": negatives_per_positive,
            "gamma": gamma,
        })

    # Object returned with logits. Given num_labels = 1, squeeze (batch_size, 1) -> (batch_size,)
    def forward(self, **inputs) -> torch.Tensor:
        outputs = self.model(**inputs)
        return outputs.logits.squeeze(-1)

    def loss(self, logits, labels):
        return self.loss_fn(logits.unsqueeze(1), labels.float().unsqueeze(1)) # BCE needs float labels

    # Unpack tokenized pair dict and binary labels
    def training_step(self, batch, batch_idx): # batch_idx required by Lightning module even if unused
        inputs, labels = batch
        logits = self(**inputs)
        loss = self.loss(logits, labels)
        self.log("train_loss", loss, prog_bar = True, on_step = True, on_epoch = True, batch_size = labels.shape[0])
        return loss    
    
    # Collect validation scores for epoch end threshold calibration
    def on_validation_epoch_start(self):
        self._val_scores = []
        self._val_targets = [] 

    # Find optimal threshold via validation predictions, then reset for next epoch
    def on_validation_epoch_end(self):
        targets = torch.cat(self._val_targets).numpy()
        scores = torch.cat(self._val_scores).numpy()

        # Reuse contrastive module's threshold calibration via F1 maximization 
        threshold = calibrate_threshold(targets, scores)
        self.eval_threshold.fill_(float(threshold))

        # Compute metrics given calibrated threshold 
        preds = (torch.cat(self._val_scores) >= threshold).int() 
        true = torch.cat(self._val_targets).int()
        self.val_acc.update(preds, true) 
        self.val_f1.update(preds, true)
        self.log("val_acc", self.val_acc.compute(), prog_bar = True)
        self.log("val_f1", self.val_f1.compute(), prog_bar = True)
        self.val_acc.reset()
        self.val_f1.reset()
    
    # Compute predictions and accumulate metrics
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self(**inputs)
        
        # Store sigmoid probabilities / scores and targets for threshold calibration
        scores = torch.sigmoid(logits)
        self._val_scores.append(scores.detach().float().cpu())
        self._val_targets.append(labels.detach().float().cpu())

        # Also log validation loss for monitoring
        loss = self.loss(logits, labels)
        self.log("val_loss", loss, prog_bar = False, on_step = False, on_epoch = True, batch_size = labels.shape[0])
    
    # Two test dataloaders: 0 = PAN21 (primary test set), 1 = PAN20
    def test_step(self, batch, batch_idx, dataloader_idx = 0):
        inputs, labels = batch
        logits = self(**inputs)
        scores = torch.sigmoid(logits)
        threshold = float(self.eval_threshold.item()) 
        preds = (scores >= threshold).int() 
        labels = labels.int()

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
    # Classifier head: higher lr since learning from scratch
    def configure_optimizers(self):
        param_groups = []

        # Encoder parameters (will be empty if frozen since requires_grad = False)
        encoder_parameters = [(n, p) for n, p in self.model.named_parameters() if "classifier" not in n and p.requires_grad]
        param_groups.extend(
            split_weight_decay_params(
                encoder_parameters,
                lr = self.hparams.lr,
                weight_decay = self.hparams.weight_decay,
            )
        )

        # Classification head parameters 
        head_lr = self.hparams.lr * self.hparams.head_lr_multiplier
        head_parameters = [(n, p) for n, p in self.model.named_parameters() if "classifier" in n and p.requires_grad]
        param_groups.extend(
            split_weight_decay_params(
                head_parameters,
                lr = head_lr,
                weight_decay = self.hparams.weight_decay,
            )
        )

        # Filter out empty groups (i.e. encoder parameters when freezing)
        param_groups = [g for g in param_groups if g["params"]]

        # AdamW optimizer with linear warmup + decay schedule
        optimizer = bnb.optim.PagedAdamW32bit(param_groups, betas=(self.beta1, self.beta2), eps=self.eps) # it's free VRAM
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
