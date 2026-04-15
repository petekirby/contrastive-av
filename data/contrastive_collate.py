import torch
from transformers import AutoTokenizer


# For simplicity, wraps around to the beginning if starting near the end.
def random_token_span(tokenizer, text, total_length, prefix=""):
    s = " " + text
    i = torch.randint(len(s), (1,)).item()
    start = s.rfind(" ", 0, i) + 1

    sample = s[start:] + s[:start - 1]
    if prefix:
        sample = prefix + sample

    return tokenizer(
        sample,
        add_special_tokens=True,
        truncation=True,
        max_length=total_length,
        padding="max_length",
    )


class ContrastiveCollator:
    def __init__(self, model_name, max_length, prefix=""):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.max_length = max_length
        self.prefix = prefix

    def __call__(self, batch):
        enc = [
            random_token_span(
                self.tokenizer,
                x["text"],
                total_length=self.max_length,
                prefix=self.prefix,
            )
            for x in batch
        ]
        enc = {k: torch.tensor([x[k] for x in enc], dtype=torch.long) for k in enc[0]}
        return enc, torch.tensor([x["author_int"] for x in batch], dtype=torch.long)


class ContrastivePairCollator:
    def __init__(self, model_name, max_length, prefix=""):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.max_length = max_length
        self.prefix = prefix

    def __call__(self, batch):
        enc1 = [
            random_token_span(
                self.tokenizer,
                x["text1"],
                total_length=self.max_length,
                prefix=self.prefix,
            )
            for x in batch
        ]
        enc2 = [
            random_token_span(
                self.tokenizer,
                x["text2"],
                total_length=self.max_length,
                prefix=self.prefix,
            )
            for x in batch
        ]
        return {
            "same": torch.tensor([int(x["same"]) for x in batch], dtype=torch.long),
            "enc1": {k: torch.tensor([x[k] for x in enc1], dtype=torch.long) for k in enc1[0]},
            "enc2": {k: torch.tensor([x[k] for x in enc2], dtype=torch.long) for k in enc2[0]},
        }
