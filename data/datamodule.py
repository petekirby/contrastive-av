import lightning as L
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader


class AuthorshipDataModule(L.LightningDataModule):
    def __init__(self, dataset_name, config_name=None, author_col="author_id", text_col="text", batch_size=32, num_workers=0):
        super().__init__()
        self.dataset_name = dataset_name
        self.config_name = config_name
        self.author_col = author_col
        self.text_col = text_col
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        ds = load_dataset(self.dataset_name, self.config_name) if self.config_name else load_dataset(self.dataset_name)
        authors = list(ds["train"][self.author_col]) + list(ds["validation"][self.author_col])
        self.author_to_int = {str(a): i for i, a in enumerate(sorted(set(map(str, authors))))}
        self.train_data = ds["train"]
        self.val_data = ds["validation"]

    def train_dataloader(self, shuffle=True):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=lambda batch: (
                torch.tensor([self.author_to_int[str(x[self.author_col])] for x in batch]),
                [str(x[self.text_col]) for x in batch],
            ),
        )


if __name__ == "__main__":
    dm = AuthorshipDataModule("peterkirby/pan2020_dict_author_fandom_doc", num_workers=6)
    dm.setup()
    labels, texts = next(iter(dm.train_dataloader()))
    print(labels.shape, len(texts), labels[0].item(), texts[0][:200])
    print(repr(texts[0][:300]))
