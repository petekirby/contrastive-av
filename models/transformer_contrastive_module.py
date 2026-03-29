import lightning.pytorch as pl
import torch
from pytorch_metric_learning.losses import BaseMetricLossFunction
from transformer_embedding_model import TransformerEmbeddingModel


# Source: https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
class TransformerContrastiveModule(pl.LightningModule):
    def __init__(
        self,
        model_config: dict,
        loss_class: type[BaseMetricLossFunction],
        loss_config: dict,
        lr: float = 2e-5,
        projection_lr_multiplier: float = 5.0,
        weight_decay: float = 0.01,
    ):
        super().__init__()
        self.model = TransformerEmbeddingModel(**model_config)
        self.loss_fn = loss_class(**loss_config)
        self.save_hyperparameters(
            {
                "model_config": model_config,
                "loss_class_name": loss_class.__name__,
                "loss_config": loss_config,
                "lr": lr,
                "projection_lr_multiplier": projection_lr_multiplier,
                "weight_decay": weight_decay,
            }
        )

    # format: {"input_ids": ..., "attention_mask": ..., sometimes "token_type_ids": ...}
    # usage: inputs = tokenizer(...), outputs = model(**inputs)
    def forward(self, **inputs) -> torch.Tensor:
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs, target)
        loss = torch.nn.functional.nll_loss(output, target.view(-1))
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            [
                {
                    "params": self.model.encoder.parameters(),
                    "lr": self.hparams.lr,
                    "weight_decay": self.hparams.weight_decay,
                },
                {
                    "params": self.model.projection.parameters(),
                    "lr": self.hparams.lr * self.hparams.projection_lr_multiplier,
                    "weight_decay": self.hparams.weight_decay,
                },
            ]
        )
