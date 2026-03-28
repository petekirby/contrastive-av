import logging
import os
import sys
from dataclasses import dataclass
import torch

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    RobertaForCL,
    BertForCL,
)

logger = logging.getLogger(__name__)

def main():
    # See all possible arguments  by passing the --help flag to this script.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, OurTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # Only one argument and it's a json file, parse it to get arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)

    # Load pretrained model and tokenizer
    config_kwargs = {"cache_dir": model_args.cache_dir, "revision": model_args.model_revision, "use_auth_token": None}
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        raise NotImplementedError

    tokenizer_kwargs = {"cache_dir": model_args.cache_dir, "use_fast": model_args.use_fast_tokenizer, "revision": model_args.model_revision, "use_auth_token": None}
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.model_name_or_path:
        if 'roberta' in model_args.model_name_or_path:
            model = RobertaForCL.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                model_args=model_args                  
            )
        elif 'bert' in model_args.model_name_or_path:
            model = BertForCL.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                model_args=model_args
            )
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    model.resize_token_embeddings(len(tokenizer))

    # Training
    if training_args.do_train:
        # train ...

    # Evaluation
    if training_args.do_eval:
        # evaluate ...

if __name__ == "__main__":
    main()
