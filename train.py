from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch import LightningModule
from data.pan_data import PANDataModule
import warnings
import torch
import wandb


class TrainCLI(LightningCLI):
    def after_fit(self):
        # Baseline model uses sklearn (no Lightning checkpoint)
        if hasattr(self.model, "is_fitted"):
            ckpt_path = None
        else:
            ckpt_path = "best"
        self.trainer.test(model=self.model, datamodule=self.datamodule, ckpt_path=ckpt_path)
        for ext in ("yaml", "csv"):
            wandb.save(f"{self.trainer.log_dir}/*.{ext}", policy="now")


def main():
    checkpoint_callback = ModelCheckpoint(monitor="val_f1", mode="max", save_top_k=1, save_last=False, filename="{epoch}-{step}-{val_f1:.4f}")
    TrainCLI(
        LightningModule,
        PANDataModule,
        subclass_mode_model=True,
        seed_everything_default=1000,
        save_config_kwargs={"overwrite": True},
        trainer_defaults={
            "default_root_dir": "output",
            "callbacks": [checkpoint_callback],
        },
    )

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.allow_unspec_int_on_nn_module = True
    torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)
    main()
