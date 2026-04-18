# This code is used to set whether transformer model parameters receive weight decay or LR multipliers or not.

def split_weight_decay_params(named_parameters, lr, weight_decay, apply_weight_decay=True):
    decay_params, no_decay_params = [], []
    for name, param in named_parameters:
        if not param.requires_grad:
            continue
        if not apply_weight_decay or param.ndim <= 1 or name.endswith(".bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    groups = []
    if decay_params:
        groups.append({"params": decay_params, "lr": lr, "weight_decay": weight_decay})
    if no_decay_params:
        groups.append({"params": no_decay_params, "lr": lr, "weight_decay": 0.0})
    return groups

def unique_named_parameters(named_parameters, seen):
    for name, param in named_parameters:
        if id(param) in seen:
            continue
        seen.add(id(param))
        yield name, param

def build_param_groups(model, base_lr, head_lr_multiplier, weight_decay, loss_fn, loss_optim_config=None):
    param_groups = []
    seen = set()

    projection_lr = base_lr * head_lr_multiplier
    param_groups.extend(
        split_weight_decay_params(
            unique_named_parameters(model.projection.named_parameters(), seen),
            lr=projection_lr,
            weight_decay=weight_decay,
        )
    )

    # shared projection/pooler params are skipped
    param_groups.extend(
        split_weight_decay_params(
            unique_named_parameters(model.encoder.named_parameters(), seen),
            lr=base_lr,
            weight_decay=weight_decay,
        )
    )

    loss_optim_config = loss_optim_config or {}
    loss_params = list(unique_named_parameters(loss_fn.named_parameters(), seen))
    if loss_params:
        loss_lr = base_lr * loss_optim_config.get("lr_multiplier", 1.0)
        param_groups.extend(
            split_weight_decay_params(
                loss_params,
                lr=loss_lr,
                weight_decay=loss_optim_config.get("weight_decay", 0.0),
                apply_weight_decay=loss_optim_config.get("apply_weight_decay", False),
            )
        )

    return param_groups
