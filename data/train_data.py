import lightning as L
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from pytorch_metric_learning.samplers import MPerClassSampler


class TrainDataModule(L.LightningDataModule):
    def __init__(self, batch_size=32, num_workers=0, sampler="random", m=2):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sampler = sampler
        self.m = m

    def setup(self, stage=None):
        self.train_data = load_dataset(
            "parquet",
            data_files="https://huggingface.co/datasets/peterkirby/pan2020_dict_author_fandom_doc/resolve/main/train.parquet",
            split="train",
        )

    def train_dataloader(self):
        if self.sampler == "mperclass":
            sampler = MPerClassSampler(
                labels=self.train_data["author_int"],
                m=self.m,
                batch_size=self.batch_size,
                length_before_new_iter=len(self.train_data),
            )
            return DataLoader(
                self.train_data,
                batch_size=self.batch_size,
                sampler=sampler,
                num_workers=self.num_workers,
                drop_last=True,
                collate_fn=lambda b: (
                    torch.tensor([x["author_int"] for x in b]),
                    [x["text"] for x in b],
                ),
            )

        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=lambda b: (
                torch.tensor([x["author_int"] for x in b]),
                [x["text"] for x in b],
            ),
        )
