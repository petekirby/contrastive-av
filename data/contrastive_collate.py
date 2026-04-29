import torch
from transformers import AutoTokenizer


# For simplicity, wraps around to the beginning if starting near the end.
def random_span_text(text, prefix="", random=False):
    if not random:
        return prefix + text

    s = " " + text
    i = torch.randint(len(s), (1,)).item()
    start = s.rfind(" ", 0, i) + 1
    sample = s[start:] + s[:start - 1]

    if prefix:
        sample = prefix + sample

    return sample


class ContrastiveCollator:
    def __init__(self, model_name, max_length, prefix="", padding_left=False, random_span=True, short_length=None, short_chance=0):
        self._lazy_tokenizer = None
        self.model_name = model_name
        self.max_length = max_length
        self.prefix = prefix
        self.padding_left = padding_left
        self.random_span = random_span
        self.short_length = short_length
        self.short_chance = short_chance

    def tokenizer(self, *args, **kwargs):
        if self._lazy_tokenizer is None:
            self._lazy_tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True, padding_side="left" if self.padding_left else "right")
        return self._lazy_tokenizer(*args, **kwargs)

    def __call__(self, batch):
        max_length = self.max_length
        if self.short_length is not None and torch.rand(()) <= self.short_chance:
            max_length = min(self.short_length, self.max_length)

        texts = [random_span_text(x["text"], prefix=self.prefix, random=self.random_span) for x in batch]

        enc = self.tokenizer(
            texts,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return dict(enc), torch.tensor([x["author_int"] for x in batch], dtype=torch.long)


class ContrastivePairCollator:
    def __init__(self, model_name, max_length, prefix="", padding_left=False):
        self._lazy_tokenizer = None
        self.model_name = model_name
        self.max_length = max_length
        self.prefix = prefix
        self.padding_left = padding_left

    def tokenizer(self, *args, **kwargs):
        if self._lazy_tokenizer is None:
            self._lazy_tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True, padding_side="left" if self.padding_left else "right")
        return self._lazy_tokenizer(*args, **kwargs)

    def __call__(self, batch):
        texts1 = [random_span_text(x["text1"], prefix=self.prefix, random=False) for x in batch]
        texts2 = [random_span_text(x["text2"], prefix=self.prefix, random=False) for x in batch]

        enc1 = self.tokenizer(
            texts1,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        enc2 = self.tokenizer(
            texts2,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "same": torch.tensor([int(x["same"]) for x in batch], dtype=torch.long),
            "enc1": dict(enc1),
            "enc2": dict(enc2),
        }
