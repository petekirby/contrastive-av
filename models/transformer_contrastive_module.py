import lightning.pytorch as pl
import torch
from loss_defaults import build_loss_fn
from optimizer_utils import build_param_groups
from transformer_embedding_model import TransformerEmbeddingModel


# Source: https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
class TransformerContrastiveModule(pl.LightningModule):
    def __init__(
        self,
        model_config: dict,
        loss_dict: dict,
        lr: float = 2e-5,
        head_lr_multiplier: float = 5.0,
        weight_decay: float = 0.01,
    ):
        super().__init__()
        self.model = TransformerEmbeddingModel(**model_config)
        self.loss_fn, self.miner, loss_dict = build_loss_fn(loss_dict)

        self.save_hyperparameters(
            {
                "model_config": model_config,
                "loss_dict": loss_dict,
                "lr": lr,
                "head_lr_multiplier": head_lr_multiplier,
                "weight_decay": weight_decay,
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
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=target.shape[0])
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
        return torch.optim.AdamW(param_groups)
