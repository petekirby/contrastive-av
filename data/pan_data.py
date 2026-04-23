# PANDataModule should be used by all models.
# Any adaptations/preprocessing needed for your model should be put in the collators.

import os
import lightning as L
from datasets import load_dataset
from torch.utils.data import DataLoader
from pytorch_metric_learning.samplers import MPerClassSampler
from .contrastive_collate import ContrastiveCollator, ContrastivePairCollator
from models.transformer_contrastive_module import TransformerContrastiveModule
from models.transformer_classification_module import TransformerClassificationModule                                                       
from .classification_collate import ClassificationCollator, ClassificationPairCollator          


class PANDataModule(L.LightningDataModule):
    def __init__(self, batch_size=64, eval_batch_size=None, sampler="mperclass", m=2, num_threads=8, max_length=512, text_prefix="", tokenizer_parallelism=False, tokenizer_name=None, padding_left=False, random_span=True):
        super().__init__()
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size or batch_size
        self.num_threads = num_threads
        self.sampler = sampler
        self.m = m
        self.max_length = max_length
        self.text_prefix = text_prefix.replace("\\n", "\n")
        self.tokenizer_parallelism = tokenizer_parallelism
        self.tokenizer_name = tokenizer_name
        self.padding_left = padding_left
        self.random_span = random_span
        self.collate_fn = None
        self.pair_collate_fn = None
        if self.tokenizer_parallelism:
            self.num_workers = 1
            os.environ["TOKENIZERS_PARALLELISM"] = "true"
            os.environ["RAYON_NUM_THREADS"] = str(self.num_threads)
        else:
            self.num_workers = self.num_threads
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def setup(self, stage=None):
        if stage in (None, "fit"):
            self.train_data = load_dataset("peterkirby/pan2020_dict_author_fandom_doc", "default", split="train")
            self.val_pairs = load_dataset("peterkirby/pan2020_dict_author_fandom_doc", "pan21", split="validation")
        if stage in (None, "test"):
            self.test_pairs = load_dataset("peterkirby/pan2020_dict_author_fandom_doc", "pan21", split="test")
            self.pan20_test_pairs = load_dataset("peterkirby/pan2020_dict_author_fandom_doc", "pan20", split="test")
        if self.collate_fn is not None and self.pair_collate_fn is not None:
            return
        model = self.trainer.lightning_module
        tokenizer_name = self.tokenizer_name or model.hparams.model_config["model_name_or_path"]
        self.max_length = min(self.max_length, model.model.config.max_position_embeddings)
        if isinstance(model, TransformerContrastiveModule):
            self.collate_fn = ContrastiveCollator(tokenizer_name, self.max_length, prefix=self.text_prefix, padding_left=self.padding_left, random_span=self.random_span)
            self.pair_collate_fn = ContrastivePairCollator(tokenizer_name, self.max_length, prefix=self.text_prefix, padding_left=self.padding_left)
        elif isinstance(model, TransformerClassificationModule):
            self.collate_fn = ClassificationCollator(tokenizer_name, self.max_length, prefix = self.text_prefix, negatives_per_positive=model.hparams.negatives_per_positive)
            self.pair_collate_fn = ClassificationPairCollator(tokenizer_name, self.max_length, prefix = self.text_prefix)
        else:
            raise ValueError(f"model unrecognized: {type(model).__name__}")

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
                persistent_workers=self.num_workers > 0,
                pin_memory=True,
                drop_last=True,
                collate_fn=self.collate_fn,
            )

        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
            drop_last=True,
            collate_fn=self.collate_fn,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_pairs,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            collate_fn=self.pair_collate_fn,
        )

    def test_dataloader(self):
        pan21_test_loader = DataLoader(
            self.test_pairs,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            collate_fn=self.pair_collate_fn,
        )
        pan20_test_loader = DataLoader(
            self.pan20_test_pairs,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            collate_fn=self.pair_collate_fn,
        )
        return [pan21_test_loader, pan20_test_loader]
