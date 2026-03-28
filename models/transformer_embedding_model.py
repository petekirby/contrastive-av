import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer


class TransformerEmbeddingModel(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        config_name: str | None = None,
        tokenizer_name: str | None = None,
        cache_dir: str | None = None,
        model_revision: str = "main",
        use_fast_tokenizer: bool = True,
        use_auth_token: bool = False,
        pooling: str = "mean",
        head_type: str = "none",
        projection_dim: int | None = None,
        projection_hidden_dim: int | None = None,
        normalize: bool = True,
        resize_token_embeddings: bool = False,
    ):
        super().__init__()

        if pooling not in {"cls", "mean", "bert_pooler"}:
            raise ValueError(f"pooling: {pooling}")

        if head_type not in {"none", "simcse", "simclr", "diffcse"}:
            raise ValueError(f"head_type: {head_type}")

        self.pooling = pooling
        self.head_type = head_type
        self.normalize = normalize

        pretrained_kwargs = {"cache_dir": cache_dir, "revision": model_revision, "use_auth_token": use_auth_token}

        self.config = AutoConfig.from_pretrained(
            config_name or model_name_or_path,
            **pretrained_kwargs,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name or model_name_or_path,
            use_fast=use_fast_tokenizer,
            **pretrained_kwargs,
        )
        self.encoder = AutoModel.from_pretrained(
            model_name_or_path,
            config=self.config,
            **pretrained_kwargs,
        )

        if resize_token_embeddings:
            self.encoder.resize_token_embeddings(len(self.tokenizer))

        hidden_size = self.config.hidden_size
        output_dim = projection_dim or hidden_size
        hidden_dim = projection_hidden_dim or hidden_size

        # Source: https://arxiv.org/abs/1908.10084 (Sentence-BERT)
        if head_type == "none":
            self.projection = nn.Identity()

        # Source: https://arxiv.org/abs/2104.08821 (SimCSE)
        elif self.pooling == "bert_pooler":
            if head_type != "simcse":
                raise ValueError("bert_pooler requires simcse")
            if not hasattr(self.encoder, "pooler") or self.encoder.pooler is None:
                raise ValueError("bert_pooler found no pooler")
            self.projection = self.encoder.pooler

        # Source: https://arxiv.org/abs/2104.08821 (SimCSE with new projection head of configurable size)
        elif head_type == "simcse":
            self.projection = nn.Sequential(
                nn.Linear(hidden_size, output_dim),
                nn.Tanh(),
            )

        # Source: https://arxiv.org/abs/2002.05709 (SimCLR)
        # Source: https://aclanthology.org/2021.icon-main.33.pdf (Contrastive Learning of Sentence Representations)
        elif head_type == "simclr":
            self.projection = nn.Sequential(
                nn.Linear(hidden_size, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )

        # Source: https://arxiv.org/abs/2204.10298 (DiffCSE)
        elif head_type == "diffcse":
            self.projection = nn.Sequential(
                nn.Linear(hidden_size, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, output_dim),
            )

        self.embedding_dim = output_dim

    @staticmethod
    # Source: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2#usage-huggingface-transformers
    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.pooling == "bert_pooler":
            return last_hidden_state
        if self.pooling == "cls":
            return last_hidden_state[:, 0]
        if attention_mask is None:
            raise ValueError("attention_mask")
        return self.mean_pooling(last_hidden_state, attention_mask)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        encoder_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            **kwargs,
        }
        if token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**encoder_kwargs)
        embeddings = self.projection(self.pool(outputs.last_hidden_state, attention_mask))
        return F.normalize(embeddings, p=2, dim=-1) if self.normalize else embeddings
