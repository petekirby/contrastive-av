from datasets import load_dataset
from torch.utils.data import DataLoader


def load_pair_dataloader(collate_fn, config_name="default", split_name="validation", batch_size=64, num_workers=0):
    ds = load_dataset(
        "peterkirby/pan2020_dict_author_fandom_doc",
        config_name,
        split=split_name,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
