import torch
from transformers import AutoTokenizer


class ContrastiveCollator:
    def __init__(self, model_name, max_length):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.max_length = max_length

    def __call__(self, batch):
        return (
            self.tokenizer(
                [x["text"] for x in batch],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ),
            torch.tensor([x["author_int"] for x in batch], dtype=torch.long),
        )


class ContrastivePairCollator:
    def __init__(self, model_name, max_length):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.max_length = max_length

    def __call__(self, batch):
        return {
            "same": torch.tensor([int(x["same"]) for x in batch], dtype=torch.long),
            "enc1": self.tokenizer(
                [x["text1"] for x in batch],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ),
            "enc2": self.tokenizer(
                [x["text2"] for x in batch],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ),
        }