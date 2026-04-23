from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch import LightningModule
from data.pan_data import PANDataModule
import warnings
import torch


class TrainCLI(LightningCLI):
    def after_fit(self):
        self.trainer.test(model=self.model, datamodule=self.datamodule, ckpt_path="best")


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
    main()
