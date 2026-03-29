import lightning.pytorch as pl
import torch
from optimizer_utils import build_param_groups
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
        head_lr_multiplier: float = 5.0,
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
                "head_lr_multiplier": head_lr_multiplier,
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
        param_groups = build_param_groups(
            model=self.model,
            base_lr=self.hparams.lr,
            head_lr_multiplier=self.hparams.head_lr_multiplier,
            weight_decay=self.hparams.weight_decay,
            loss_fn=self.loss_fn,
            loss_optim_config=self.hparams.loss_optim_config,
        )
        return torch.optim.AdamW(param_groups)
