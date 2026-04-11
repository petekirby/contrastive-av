from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch import LightningModule

from data.pan_data import PANDataModule


class TrainCLI(LightningCLI):
    def after_fit(self):
        checkpoint_callback = next(cb for cb in self.trainer.callbacks if isinstance(cb, ModelCheckpoint))
        best_checkpoint_path = checkpoint_callback.best_model_path or checkpoint_callback.last_model_path
        best_model = self.model.__class__.load_from_checkpoint(best_checkpoint_path)
        self.trainer.validate(model=best_model, datamodule=self.datamodule, ckpt_path=None)
        self.trainer.test(model=best_model, datamodule=self.datamodule, ckpt_path=None)


def main():
    TrainCLI(
        LightningModule,
        PANDataModule,
        subclass_mode_model=True,
        seed_everything_default=42,
        save_config_kwargs={"overwrite": True},
        trainer_defaults={"default_root_dir": "output"},
    )

if __name__ == "__main__":
    main()