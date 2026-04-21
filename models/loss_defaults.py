# Only contrastive objectives need these loss functions.

from copy import deepcopy

from pytorch_metric_learning import distances, losses, miners


def merge_dicts(result, override):
    if not override:
        return result
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


DISTANCE_CONFIGS = {
    "cosine": distances.CosineSimilarity,
    "l2": lambda: distances.LpDistance(normalize_embeddings=True, p=2, power=1),
}


LOSS_CONFIGS = {
    "ntxent": {
        "loss_class": losses.NTXentLoss,
        "loss_config": {
            "temperature": 0.05,
        },
        "loss_optim_config": {},
    },
    "supcon": {
        "loss_class": losses.SupConLoss,
        "loss_config": {
            "temperature": 0.05,
        },
        "loss_optim_config": {},
    },
    "contrastive": {
        "loss_class": losses.ContrastiveLoss,
        "miner": miners.BatchEasyHardMiner,
        "loss_config": {
            "pos_margin": 1.0,
            "neg_margin": 0.5,
            "distance": "cosine",
        },
        "loss_optim_config": {},
    },
    "multisimilarity": {
        "loss_class": losses.MultiSimilarityLoss,
        "loss_config": {
            "alpha": 10,
            "beta": 50,
            "base": 0.5,
        },
        "loss_optim_config": {},
    },
    "circle": {
        "loss_class": losses.CircleLoss,
        "loss_config": {
            "m": 0.4,
            "gamma": 80,
            "distance": "cosine",
        },
        "loss_optim_config": {},
    },
    "generalized_lifted": {
        "loss_class": losses.GeneralizedLiftedStructureLoss,
        "loss_config": {
            "pos_margin": 0.0,
            "neg_margin": 1.0,
            "distance": "l2",
        },
        "loss_optim_config": {},
    },
    "proxyanchor": {
        "loss_class": losses.ProxyAnchorLoss,
        "loss_config": {
            "margin": 0.1,
            "alpha": 32,
        },
        "loss_optim_config": {
            "lr_multiplier": 100.0,
            "weight_decay": 0.0,
            "apply_weight_decay": False,
        },
    },
    "softtriple": {
        "loss_class": losses.SoftTripleLoss,
        "loss_config": {
            "centers_per_class": 10,
            "la": 20,
            "gamma": 0.1,
            "margin": 0.01,
        },
        "loss_optim_config": {
            "lr_multiplier": 100.0,
            "weight_decay": 0.0,
            "apply_weight_decay": False,
        },
    },
}


# format: {"name": "...", "loss_config": {...}, optional "loss_optim_config": {...}}
def build_loss_fn(loss_dict):
    defaults = deepcopy(LOSS_CONFIGS[loss_dict["name"].lower()])
    config = {
        "name": loss_dict["name"],
        "loss_config": merge_dicts(defaults["loss_config"], loss_dict.get("loss_config")),
        "loss_optim_config": merge_dicts(defaults["loss_optim_config"], loss_dict.get("loss_optim_config")),
    }

    loss_class = defaults["loss_class"]
    loss_kwargs = deepcopy(config["loss_config"])
    if "distance" in loss_kwargs:
        loss_kwargs["distance"] = DISTANCE_CONFIGS[loss_kwargs["distance"]]()
    miner = defaults["miner"](distance=loss_kwargs["distance"]) if "miner" in defaults else None
    return loss_class(**loss_kwargs), miner, config
