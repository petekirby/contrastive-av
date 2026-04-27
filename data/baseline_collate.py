# Responsibility: Henry

import torch
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu


# For simplicity, wraps around to the beginning if starting near the end
def random_span_text(text, prefix="", random=False, max_chars=None):
    if not random:
        sample = text
    else:
        s = " " + text
        i = torch.randint(len(s), (1,)).item()
        start = s.rfind(" ", 0, i) + 1
        sample = s[start:] + s[:start - 1]

    if prefix:
        sample = prefix + sample

    if max_chars is not None:
        sample = sample[:max_chars]

    return sample


class BaselineCollator:
    def __init__(self, prefix="", random_span=True, negatives_per_positive=1, max_chars=None):
        self.prefix = prefix
        self.random_span = random_span
        self.negatives_per_positive = negatives_per_positive
        self.max_chars = max_chars

    # Batch from MPerClassSampler, each item consists of 'text' and 'author_int'
    # Returns raw text pairs for TF-IDF feature extraction instead of tokenized tensors
    def __call__(self, batch):
        texts = [x["text"] for x in batch]
        author_labels = torch.tensor([x["author_int"] for x in batch], dtype=torch.long)

        # Triplet mining gives anchor/positive/negative indices for pair construction
        if self.negatives_per_positive >= len(author_labels) - 2:
            a, p, n = lmu.convert_to_triplets(None, author_labels, t_per_anchor="all")
        else:
            a, p, n = lmu.get_random_triplet_indices(author_labels, t_per_anchor=self.negatives_per_positive)

        # Build positive pairs (same author, label = 1) and negative pairs (different author, label = 0)
        pair_texts1 = []
        pair_texts2 = []
        pair_labels = []
        seen_pos = set()

        for i in range(len(a)):
            a_idx, p_idx, n_idx = a[i].item(), p[i].item(), n[i].item()
            text_a = random_span_text(texts[a_idx], prefix=self.prefix, random=self.random_span, max_chars=self.max_chars)

            if (a_idx, p_idx) not in seen_pos:
                seen_pos.add((a_idx, p_idx))
                pair_texts1.append(text_a)
                pair_texts2.append(random_span_text(texts[p_idx], prefix=self.prefix, random=self.random_span, max_chars=self.max_chars))
                pair_labels.append(1)

            # Add one negative pair for each sampled triplet
            pair_texts1.append(text_a)
            pair_texts2.append(random_span_text(texts[n_idx], prefix=self.prefix, random=self.random_span, max_chars=self.max_chars))
            pair_labels.append(0)

        # inputs dict, labels tensor
        return {
            "text1": pair_texts1,
            "text2": pair_texts2,
        }, torch.tensor(pair_labels, dtype=torch.long)


class BaselinePairCollator:
    def __init__(self, prefix="", max_chars=None):
        self.prefix = prefix
        self.max_chars = max_chars

    # Collates pre-constructed validation/test pairs and returns raw text
    def __call__(self, batch):
        texts1 = [random_span_text(x["text1"], prefix=self.prefix, random=False, max_chars=self.max_chars) for x in batch]
        texts2 = [random_span_text(x["text2"], prefix=self.prefix, random=False, max_chars=self.max_chars) for x in batch]

        # inputs dict, labels tensor
        return {
            "text1": texts1,
            "text2": texts2,
        }, torch.tensor([int(x["same"]) for x in batch], dtype=torch.long)
