import torch
from transformers import AutoTokenizer


def contrastive_collate_fn(model_name, max_length):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def collate_fn(batch):
        return (
            tokenizer(
                [x["text"] for x in batch],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ),
            torch.tensor([x["author_int"] for x in batch], dtype=torch.long),
        )

    return collate_fn


def contrastive_pair_collate_fn(model_name, max_length):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def collate_fn(batch):
        return {
            "same": torch.tensor([int(x["same"]) for x in batch], dtype=torch.long),
            "enc1": tokenizer(
                [x["text1"] for x in batch],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ),
            "enc2": tokenizer(
                [x["text2"] for x in batch],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ),
        }

    return collate_fn
