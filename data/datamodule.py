import lightning as L
import torch
from collections import defaultdict
from datasets import load_dataset
from torch.utils.data import DataLoader


class AuthorshipDataModule(L.LightningDataModule):
    def __init__(self, dataset_name, config_name=None, author_col="author_int", text_col="text", fandom_col="fandom_int", batch_size=32, num_workers=0):
        super().__init__()
        self.dataset_name = dataset_name
        self.config_name = config_name
        self.author_col = author_col
        self.text_col = text_col
        self.fandom_col = fandom_col
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        train = load_dataset(
            "parquet",
            data_files=f"https://huggingface.co/datasets/{self.dataset_name}/resolve/main/train.parquet",
            split="train",
        )
        validation = load_dataset(
            "parquet",
            data_files=f"https://huggingface.co/datasets/{self.dataset_name}/resolve/main/validation.parquet",
            split="train",
        )
        authors = list(train[self.author_col]) + list(validation["author1_int"]) + list(validation["author2_int"])
        self.author_to_int = {str(a): i for i, a in enumerate(sorted(set(map(str, authors))))}
        self.train_data = train
        self.val_data = validation

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

    val_authors = [str(a) for a in dm.val_data["author1_int"]] + [str(a) for a in dm.val_data["author2_int"]]
    distinct_val_authors = len(set(val_authors))
    valid_da_count = distinct_val_authors // 2

    author_to_fandoms = defaultdict(set)
    for author, fandom in zip(dm.val_data["author1_int"], dm.val_data["fandom1_int"]):
        author_to_fandoms[str(author)].add(str(fandom))
    for author, fandom in zip(dm.val_data["author2_int"], dm.val_data["fandom2_int"]):
        author_to_fandoms[str(author)].add(str(fandom))

    fandom_bucket_counts = {k: 0 for k in range(2, 7)}
    for fandom_count in (len(fandoms) for fandoms in author_to_fandoms.values()):
        if fandom_count in fandom_bucket_counts:
            fandom_bucket_counts[fandom_count] += 1

    valid_sa_count = sum((k * (k - 1) // 2) * bucket_count for k, bucket_count in fandom_bucket_counts.items())

    print(f"Distinct authors in validation: {distinct_val_authors}")
    print(f"Valid DA count (distinct authors / 2): {valid_da_count}")
    print(f"Distinct fandoms counted with column: author1_int/author2_int + fandom1_int/fandom2_int")
    print("Author bucket counts by distinct fandom count:")
    for k in range(2, 7):
        print(f"  fandom_count={k}: {fandom_bucket_counts[k]}")
    print(f"Valid SA count: {valid_sa_count}")