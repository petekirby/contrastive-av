import lightning.pytorch as pl
import torch
from .loss_defaults import build_loss_fn
from .optimizer_utils import build_param_groups
from .transformer_embedding_model import TransformerEmbeddingModel
from helper_functions.contrastive_eval import contrastive_evaluate


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
        self.eval_threshold = None

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

    def validation_step(self, batch, batch_idx):
        return None

    def on_validation_epoch_end(self):
        dataloader = self.trainer.val_dataloaders
        if isinstance(dataloader, (list, tuple)):
            dataloader = dataloader[0]

        threshold, acc, f1 = contrastive_evaluate(self, dataloader, threshold=None, device=self.device)
        self.eval_threshold = threshold

        self.log("val_acc", acc, prog_bar=True, on_epoch=True)
        self.log("val_f1", f1, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return None

    def on_test_epoch_end(self):
        if self.eval_threshold is None:
            raise RuntimeError("eval_threshold is None")

        dataloaders = self.trainer.test_dataloaders

        _, acc, f1 = contrastive_evaluate(self, dataloaders[0], threshold=self.eval_threshold, device=self.device)
        self.log("test_acc", acc, prog_bar=True, on_epoch=True)
        self.log("test_f1", f1, prog_bar=True, on_epoch=True)

        _, acc, f1 = contrastive_evaluate(self, dataloaders[1], threshold=self.eval_threshold, device=self.device)
        self.log("pan20_test_acc", acc, prog_bar=False, on_epoch=True)
        self.log("pan20_test_f1", f1, prog_bar=False, on_epoch=True)
