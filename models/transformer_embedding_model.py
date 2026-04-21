# This is an "embedding model" but you could just reuse it and put a linear classification head on it to make a classification model.
# If so, the module file is a fine place to extend the model that way.

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
from .torch_utils import Residual


class TransformerEmbeddingModel(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        config_name: str | None = None,
        cache_dir: str | None = None,
        model_revision: str = "main",
        use_bf16: bool = False,
        use_liger_kernel: bool = False,
        gradient_checkpointing: bool = False,
        pooling: str = "mean",
        concat_layers: int = 4,
        head_type: str = "none",
        projection_dim: int | None = None,
        projection_hidden_dim: int | None = None,
        normalize: bool = True,
        attn_implementation: str | None = None,
    ):
        super().__init__()
        self.apply_liger_kernel(model_name_or_path, use_liger_kernel)

        if pooling not in {"cls", "mean", "max", "mean_first_last", "mean_first_last_concat", "bert_pooler", "cls_concat", "last"}:
            raise ValueError(f"pooling: {pooling}")

        if head_type not in {"none", "simcse", "simclr", "diffcse", "byol", "ln_gelu_residual", "two_linear_layer"}:
            raise ValueError(f"head_type: {head_type}")

        self.pooling = pooling
        self.head_type = head_type
        self.normalize = normalize

        pretrained_kwargs = {"cache_dir": cache_dir, "revision": model_revision}

        self.config = AutoConfig.from_pretrained(
            config_name or model_name_or_path,
            **pretrained_kwargs,
        )

        pooled_size = self.config.hidden_size
        if self.pooling == "mean_first_last_concat":
            pooled_size *= 2
            self.config.output_hidden_states = True
        elif self.pooling == "mean_first_last":
            self.config.output_hidden_states = True
        elif self.pooling == "cls_concat":
            pooled_size *= concat_layers
            self.concat_layers = concat_layers
            self.cls_layer_norm = nn.LayerNorm(pooled_size)
            self.config.output_hidden_states = True

        self.encoder = AutoModel.from_pretrained(
            model_name_or_path,
            config=self.config,
            attn_implementation=attn_implementation,
            dtype=torch.bfloat16 if use_bf16 else None,
            **pretrained_kwargs,
        )

        if gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()

        if hasattr(self.encoder.config, "use_cache"):
            self.encoder.config.use_cache = False

        output_dim = projection_dim or pooled_size
        hidden_dim = projection_hidden_dim or pooled_size * 4

        # Source: https://arxiv.org/abs/2104.08821 (SimCSE)
        if self.pooling == "bert_pooler":
            if head_type != "simcse":
                raise ValueError("bert_pooler requires simcse")
            if not hasattr(self.encoder, "pooler") or self.encoder.pooler is None:
                raise ValueError("bert_pooler found no pooler")
            output_dim = pooled_size # has to be same size
            self.projection = self.encoder.pooler

        # Source: https://arxiv.org/abs/2104.08821 (SimCSE with new BERT-style projection head)
        # Source: https://github.com/princeton-nlp/SimCSE/blob/main/simcse/models.py
        elif head_type == "simcse":
            self.projection = nn.Sequential(
                nn.Linear(pooled_size, output_dim),
                nn.Tanh(),
            )

        # Source: https://arxiv.org/abs/1908.10084 (becomes Sentence-BERT with "mean" pooling)
        elif head_type == "none":
            self.projection = nn.Identity()

        # Source: https://arxiv.org/abs/2002.05709 (SimCLR)
        # Source: https://github.com/google-research/simclr/blob/master/model_util.py
        elif head_type == "simclr":
            self.projection = nn.Sequential(
                nn.Linear(pooled_size, hidden_dim, bias=True),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, output_dim, bias=False),
                nn.BatchNorm1d(output_dim),
            )

        # Source: https://arxiv.org/abs/2204.10298 (DiffCSE)
        # Source: https://github.com/voidism/DiffCSE/blob/master/diffcse/models.py
        elif head_type == "diffcse":
            self.projection = nn.Sequential(
                nn.Linear(pooled_size, hidden_dim, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, output_dim, bias=False),
                nn.BatchNorm1d(output_dim, affine=False),
            )

        # Source: https://arxiv.org/pdf/2006.07733 (BYOL: "Contrary to SimCLR, the output of this MLP is not batch normalized")
        # Source: https://github.com/google-deepmind/deepmind-research/blob/master/byol/utils/networks.py
        elif head_type == "byol":
            self.projection = nn.Sequential(
                nn.Linear(pooled_size, hidden_dim, bias=True),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, output_dim, bias=False),
            )

        # switch to layer-norm and GELU in the MLP, plus a residual
        elif head_type == "ln_gelu_residual":
            output_dim = pooled_size # has to be same size
            self.projection = Residual(nn.Sequential(
                nn.Linear(pooled_size, hidden_dim, bias=True),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, output_dim, bias=False),
            ))

        # just use two linear layers
        elif head_type == "two_linear_layer":
            self.projection = nn.Sequential(
                nn.Linear(pooled_size, hidden_dim, bias=False),
                nn.Linear(hidden_dim, output_dim, bias=False),
            )

        self.embedding_dim = output_dim

    @staticmethod
    def apply_liger_kernel(model_name_or_path, use_liger_kernel):
        if not use_liger_kernel:
            return
        name = model_name_or_path.lower()
        if "harrier-oss-v1-270m" in name or "gemma3" in name:
            from liger_kernel.transformers import apply_liger_kernel_to_gemma3_text
            apply_liger_kernel_to_gemma3_text()
        elif "harrier-oss-v1-0.6b" in name or "qwen3" in name:
            from liger_kernel.transformers import apply_liger_kernel_to_qwen3
            apply_liger_kernel_to_qwen3()
        else:
            raise ValueError(f"No liger kernel for {model_name_or_path}")

    @staticmethod
    # Source: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2#usage-huggingface-transformers
    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @staticmethod
    # Source: https://github.com/huggingface/sentence-transformers/blob/main/sentence_transformers/sentence_transformer/modules/pooling.py
    def max_pooling(token_embeddings, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand_as(token_embeddings).to(token_embeddings.dtype)
        return token_embeddings.masked_fill(mask == 0, torch.finfo(token_embeddings.dtype).min).max(dim=1).values

    @staticmethod
    # Source: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B#transformers-usage
    def last_token_pooling(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def cls_concat_pooling(self, hidden_states):
        cls_layers = [layer[:, 0, :] for layer in hidden_states[-self.concat_layers:]]
        return self.cls_layer_norm(torch.cat(cls_layers, dim=-1))

    # Source: https://github.com/princeton-nlp/SimCSE/blob/main/simcse/models.py
    def mean_first_last_pooling(self, hidden_states, attention_mask: torch.Tensor) -> torch.Tensor:
        first = self.mean_pooling(hidden_states[1], attention_mask)
        last = self.mean_pooling(hidden_states[-1], attention_mask)
        if self.pooling == "mean_first_last_concat":
            return torch.cat([first, last], dim=-1)
        return (first + last) / 2.0

    def pool(self, outputs, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        last_hidden_state: torch.Tensor = outputs.last_hidden_state
        if self.pooling == "bert_pooler":
            return last_hidden_state # used by self.encoder.pooler
        if self.pooling == "cls":
            return last_hidden_state[:, 0]
        if self.pooling == "cls_concat":
            return self.cls_concat_pooling(outputs.hidden_states)
        if attention_mask is None:
            raise ValueError("attention_mask")
        if self.pooling == "last":
            return self.last_token_pooling(last_hidden_state, attention_mask)
        if self.pooling == "max":
            return self.max_pooling(last_hidden_state, attention_mask)
        if self.pooling == "mean_first_last" or self.pooling == "mean_first_last_concat":
            return self.mean_first_last_pooling(outputs.hidden_states, attention_mask)
        return self.mean_pooling(last_hidden_state, attention_mask)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        encoder_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask, **kwargs}
        if token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = token_type_ids
        if self.pooling in {"cls_concat", "mean_first_last", "mean_first_last_concat"}:
            encoder_kwargs["output_hidden_states"] = True
        outputs = self.encoder(**encoder_kwargs)
        embeddings = self.projection(self.pool(outputs, attention_mask))
        return F.normalize(embeddings, p=2, dim=-1) if self.normalize else embeddings
