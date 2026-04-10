import torch
from transformers import AutoTokenizer


def contrastive_train_collate_fn(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def collate_fn(batch):
        return (
            tokenizer(
                [x["text"] for x in batch],
                padding=True,
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            ),
            torch.tensor([x["author_int"] for x in batch], dtype=torch.long),
        )

    return collate_fn
