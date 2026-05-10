#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score
from transformers import AutoTokenizer
from datasets import load_dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.transformer_contrastive_module import TransformerContrastiveModule  # noqa: E402


def get_init_args(cfg, key):
    section = cfg.get(key, {})
    if isinstance(section, dict) and "init_args" in section:
        return section["init_args"]
    return section


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_truth(path):
    truth = {}
    for d in read_jsonl(path):
        truth[d["id"]] = int(bool(d["same"]))
    return truth


def get_pair_texts(d):
    if "pair" in d:
        return d["pair"][0], d["pair"][1]
    if "text1" in d and "text2" in d:
        return d["text1"], d["text2"]
    raise KeyError(f"Cannot find pair texts in item with keys: {list(d.keys())}")


def load_pairs_jsonl(pairs_path, truth_path=None, require_labels=False):
    truth = load_truth(truth_path) if truth_path else None

    ids, text1, text2, labels = [], [], [], []

    for i, d in enumerate(read_jsonl(pairs_path)):
        pid = d.get("id", f"jsonl-{i:08d}")
        a, b = get_pair_texts(d)

        ids.append(pid)
        text1.append(a)
        text2.append(b)

        if truth is not None:
            labels.append(truth[pid])
        elif "same" in d:
            labels.append(int(bool(d["same"])))

    if require_labels and not labels:
        raise ValueError(
            "Validation labels are required. Provide --val-truth or use validation JSONL with a `same` field."
        )

    if labels:
        return ids, text1, text2, np.array(labels, dtype=np.int64)

    return ids, text1, text2, None


def load_pairs_hf(dataset_name, dataset_config, split):
    ds = load_dataset(dataset_name, dataset_config, split=split)

    ids, text1, text2, labels = [], [], [], []

    for i, d in enumerate(ds):
        pid = d.get("id", f"hf-{split}-{i:08d}")

        if "text1" not in d or "text2" not in d:
            raise KeyError(
                f"HF row does not contain text1/text2. Available keys: {list(d.keys())}"
            )

        if "same" not in d:
            raise KeyError(
                f"HF validation row does not contain same label. Available keys: {list(d.keys())}"
            )

        ids.append(pid)
        text1.append(d["text1"])
        text2.append(d["text2"])
        labels.append(int(bool(d["same"])))

    return ids, text1, text2, np.array(labels, dtype=np.int64)


def tokenize(tokenizer, texts, max_length, prefix, device):
    texts = [prefix + t for t in texts]

    enc = tokenizer(
        texts,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )

    return {k: v.to(device) for k, v in enc.items()}


def score_pairs(
    model,
    tokenizer,
    ids,
    text1,
    text2,
    *,
    max_length,
    prefix,
    batch_size,
    device,
):
    model.eval()
    scores = []
    n = len(ids)

    with torch.inference_mode():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)

            enc1 = tokenize(tokenizer, text1[start:end], max_length, prefix, device)
            enc2 = tokenize(tokenizer, text2[start:end], max_length, prefix, device)

            emb1 = model(**enc1)
            emb2 = model(**enc2)

            sims = F.cosine_similarity(emb1, emb2).detach().float().cpu().numpy()
            scores.extend(sims.tolist())

            if start == 0 or end == n or end % 1000 == 0:
                print(f"scored {end}/{n}")

    return np.array(scores, dtype=np.float64)


def binarize_keep_abstain(values, threshold=0.5):
    values = np.asarray(values, dtype=np.float64)
    out = values.copy()

    out[values > threshold] = 1.0
    out[values < threshold] = 0.0
    out[values == threshold] = 0.5

    return out


def pan_auc(y_true, y_pred):
    try:
        return float(roc_auc_score(y_true, y_pred))
    except ValueError:
        return 0.0


def pan_c_at_1(y_true, y_pred, threshold=0.5):
    n = float(len(y_pred))
    nc, nu = 0.0, 0.0

    for gt, pred in zip(y_true, y_pred):
        if pred == threshold:
            nu += 1.0
        elif (pred > threshold) == (gt > threshold):
            nc += 1.0

    return float((1.0 / n) * (nc + (nu * nc / n)))


def pan_f1(y_true, y_pred, threshold=0.5):
    yt, yp = [], []

    for true, pred in zip(y_true, y_pred):
        if pred != threshold:
            yt.append(true)
            yp.append(1 if pred > threshold else 0)

    if not yt:
        return 0.0

    return float(f1_score(yt, yp))


def pan_f_05_u(y_true, y_pred, pos_label=1, threshold=0.5):
    pred_y = binarize_keep_abstain(y_pred, threshold=threshold)

    n_tp, n_fn, n_fp, n_u = 0, 0, 0, 0

    for true, pred in zip(y_true, pred_y):
        if pred == threshold:
            n_u += 1
        elif pred == pos_label and pred == true:
            n_tp += 1
        elif pred == pos_label and pred != true:
            n_fp += 1
        elif true == pos_label and pred != true:
            n_fn += 1

    denom = 1.25 * n_tp + 0.25 * (n_fn + n_u) + n_fp

    if denom == 0:
        return 0.0

    return float((1.25 * n_tp) / denom)


def pan_brier(y_true, y_pred):
    try:
        return float(1.0 - brier_score_loss(y_true, y_pred))
    except ValueError:
        return 0.0


def pan_metrics(y_true, y_pred):
    out = {
        "auc": pan_auc(y_true, y_pred),
        "c@1": pan_c_at_1(y_true, y_pred),
        "F1": pan_f1(y_true, y_pred),
        "f_05_u": pan_f_05_u(y_true, y_pred),
        "brier": pan_brier(y_true, y_pred),
    }

    out["overall"] = float(np.mean(list(out.values())))
    return out


def apply_abstention(probs, delta):
    probs = np.asarray(probs, dtype=np.float64)
    out = probs.copy()
    out[np.abs(out - 0.5) <= delta] = 0.5
    return out


def parse_delta_grid(s):
    if s:
        return [float(x) for x in s.split(",")]

    return [
        0.0,
        0.005,
        0.01,
        0.015,
        0.02,
        0.03,
        0.04,
        0.05,
        0.075,
        0.10,
        0.125,
        0.15,
        0.20,
    ]


def write_answers(path, ids, values):
    with open(path, "w", encoding="utf-8") as f:
        for pid, value in zip(ids, values):
            value = float(np.clip(value, 0.0, 1.0))
            f.write(json.dumps({"id": pid, "value": value}) + "\n")


def write_scores(path, ids, scores, labels=None):
    with open(path, "w", encoding="utf-8") as f:
        for i, pid in enumerate(ids):
            d = {
                "id": pid,
                "score": float(scores[i]),
            }

            if labels is not None:
                d["same"] = bool(labels[i])

            f.write(json.dumps(d) + "\n")


def load_model(args, model_config):
    """
    Load Lightning checkpoint, including checkpoints saved from torch.compile().
    torch.compile wraps the model under `_orig_mod`, causing keys like:
        model._orig_mod.encoder...
    but the normal model expects:
        model.encoder...
    """
    import tempfile
    import torch

    ckpt = torch.load(args.ckpt, map_location="cpu")

    state_dict = ckpt.get("state_dict", ckpt)

    fixed_state_dict = {}
    changed = 0

    for k, v in state_dict.items():
        new_k = k.replace("model._orig_mod.", "model.")
        if new_k != k:
            changed += 1
        fixed_state_dict[new_k] = v

    if changed:
        print(f"normalized {changed} torch.compile checkpoint keys: model._orig_mod. -> model.")
        ckpt["state_dict"] = fixed_state_dict

        with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as tmp:
            fixed_ckpt_path = tmp.name

        torch.save(ckpt, fixed_ckpt_path)
        ckpt_path = fixed_ckpt_path
    else:
        ckpt_path = args.ckpt

    try:
        return TransformerContrastiveModule.load_from_checkpoint(
            ckpt_path,
            map_location="cpu",
            model_config=model_config,
            compile=False,
        )
    except TypeError:
        try:
            return TransformerContrastiveModule.load_from_checkpoint(
                ckpt_path,
                map_location="cpu",
                model_config=model_config,
            )
        except TypeError:
            return TransformerContrastiveModule.load_from_checkpoint(
                ckpt_path,
                map_location="cpu",
            )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)

    # Validation can come from HF or local JSONL.
    ap.add_argument(
        "--val-source",
        choices=["hf", "jsonl"],
        default="hf",
        help="Use Hugging Face validation by default, or local JSONL validation.",
    )
    ap.add_argument("--val-pairs", default=None)
    ap.add_argument("--val-truth", default=None)

    ap.add_argument(
        "--hf-dataset",
        default="peterkirby/pan2020_dict_author_fandom_doc",
    )
    ap.add_argument("--hf-config", default="pan21")
    ap.add_argument("--hf-val-split", default="validation")

    # Test is your official PAN21 JSONL.
    ap.add_argument("--test-pairs", required=True)

    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-length", type=int, default=None)

    ap.add_argument(
        "--delta-grid",
        default=None,
        help="Comma-separated abstention deltas. Default uses a reasonable grid.",
    )

    ap.add_argument(
        "--attn-implementation",
        default=None,
        help="Optional override, e.g. eager, sdpa, flash_attention_2.",
    )

    ap.add_argument(
        "--platt-c",
        type=float,
        default=1.0,
        help="LogisticRegression C for Platt scaling.",
    )

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_init = get_init_args(cfg, "model")
    data_init = get_init_args(cfg, "data")

    model_config = dict(model_init["model_config"])

    if args.attn_implementation:
        model_config["attn_implementation"] = args.attn_implementation

    tokenizer_name = data_init.get("tokenizer_name") or model_config["model_name_or_path"]
    prefix = data_init.get("text_prefix", "").replace("\\n", "\n")
    padding_left = bool(data_init.get("padding_left", False))

    print("loading model...")
    model = load_model(args, model_config)
    model.to(args.device)
    model.eval()

    max_length = args.max_length or int(data_init.get("max_length", 4096))

    if hasattr(model.model, "config") and hasattr(model.model.config, "max_position_embeddings"):
        max_length = min(max_length, int(model.model.config.max_position_embeddings))

    print(f"tokenizer: {tokenizer_name}")
    print(f"max_length: {max_length}")

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_fast=True,
        padding_side="left" if padding_left else "right",
    )

    print("loading validation pairs...")

    if args.val_source == "hf":
        val_ids, val_t1, val_t2, val_y = load_pairs_hf(
            args.hf_dataset,
            args.hf_config,
            args.hf_val_split,
        )
    else:
        if not args.val_pairs:
            raise ValueError("--val-pairs is required when --val-source jsonl")

        val_ids, val_t1, val_t2, val_y = load_pairs_jsonl(
            args.val_pairs,
            args.val_truth,
            require_labels=True,
        )

    print(f"validation pairs: {len(val_ids)}")

    print("loading test pairs...")
    test_ids, test_t1, test_t2, _ = load_pairs_jsonl(
        args.test_pairs,
        truth_path=None,
        require_labels=False,
    )
    print(f"test pairs: {len(test_ids)}")

    print("scoring validation...")
    val_scores = score_pairs(
        model,
        tokenizer,
        val_ids,
        val_t1,
        val_t2,
        max_length=max_length,
        prefix=prefix,
        batch_size=args.batch_size,
        device=args.device,
    )

    print("scoring test...")
    test_scores = score_pairs(
        model,
        tokenizer,
        test_ids,
        test_t1,
        test_t2,
        max_length=max_length,
        prefix=prefix,
        batch_size=args.batch_size,
        device=args.device,
    )

    write_scores(out_dir / "val_scores.jsonl", val_ids, val_scores, val_y)
    write_scores(out_dir / "test_scores.jsonl", test_ids, test_scores, None)

    print("fitting Platt/logistic calibration on validation...")

    calibrator = LogisticRegression(
        C=args.platt_c,
        solver="lbfgs",
        max_iter=1000,
    )

    calibrator.fit(val_scores.reshape(-1, 1), val_y)

    a = float(calibrator.coef_[0, 0])
    b = float(calibrator.intercept_[0])

    decision_cosine_threshold = float(-b / a) if a != 0 else None

    val_probs = expit(a * val_scores + b)
    test_probs = expit(a * test_scores + b)

    # 1. Hard 0/1 baseline using the Platt p=0.5 cosine threshold.
    if a == 0:
        hard_threshold = 0.5
        print("WARNING: Platt slope a == 0; falling back to hard threshold 0.5")
    else:
        hard_threshold = -b / a

    if a >= 0:
        hard_val_values = (val_scores >= hard_threshold).astype(float)
    else:
        hard_val_values = (val_scores <= hard_threshold).astype(float)

    hard_metrics = pan_metrics(val_y, hard_val_values)

    # 2. Platt probabilities, no abstention.
    platt_metrics = pan_metrics(val_y, val_probs)

    # 3. Platt probabilities + validation-selected abstention.
    print("selecting abstention band on validation...")

    best = None
    rows = []

    for delta in parse_delta_grid(args.delta_grid):
        val_pred = apply_abstention(val_probs, delta)
        metrics = pan_metrics(val_y, val_pred)
        nonanswer_rate = float(np.mean(val_pred == 0.5))

        row = {
            "delta": float(delta),
            "nonanswer_rate": nonanswer_rate,
            **metrics,
        }

        rows.append(row)

        if best is None:
            best = row
        elif row["overall"] > best["overall"] + 1e-12:
            best = row
        elif abs(row["overall"] - best["overall"]) <= 1e-12 and row["delta"] < best["delta"]:
            best = row

    best_delta = float(best["delta"])

    val_values = apply_abstention(val_probs, best_delta)
    test_values = apply_abstention(test_probs, best_delta)

    platt_abstain_metrics = pan_metrics(val_y, val_values)

    write_answers(out_dir / "answers.jsonl", test_ids, test_values)
    write_answers(out_dir / "val_answers.jsonl", val_ids, val_values)

    metadata = {
        "config": args.config,
        "ckpt": args.ckpt,
        "tokenizer_name": tokenizer_name,
        "max_length": max_length,
        "batch_size": args.batch_size,
        "validation_source": args.val_source,
        "hf_dataset": args.hf_dataset if args.val_source == "hf" else None,
        "hf_config": args.hf_config if args.val_source == "hf" else None,
        "hf_val_split": args.hf_val_split if args.val_source == "hf" else None,
        "platt": {
            "formula": "p_same = sigmoid(a * cosine + b)",
            "a": a,
            "b": b,
            "decision_cosine_threshold_for_p_0_5": decision_cosine_threshold,
            "C": args.platt_c,
        },
        "validation_comparison": {
            "hard_0_1": {
                "threshold": float(hard_threshold),
                "metrics": hard_metrics,
                "overall": hard_metrics["overall"],
            },
            "platt_no_abstention": {
                "metrics": platt_metrics,
                "overall": platt_metrics["overall"],
            },
            "platt_with_abstention": {
                "delta": best_delta,
                "metrics": platt_abstain_metrics,
                "overall": platt_abstain_metrics["overall"],
                "nonanswer_rate": float(np.mean(val_values == 0.5)),
            },
        },
        "selected_abstention": best,
        "all_delta_results": rows,
        "validation_metrics_selected": platt_abstain_metrics,
        "validation_nonanswer_rate_selected": float(np.mean(val_values == 0.5)),
        "test_nonanswer_rate": float(np.mean(test_values == 0.5)),
        "num_validation_pairs": len(val_ids),
        "num_test_pairs": len(test_ids),
    }

    with open(out_dir / "calibration_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    print()
    print(f"wrote: {out_dir / 'answers.jsonl'}")
    print(f"wrote: {out_dir / 'val_answers.jsonl'}")
    print(f"wrote: {out_dir / 'val_scores.jsonl'}")
    print(f"wrote: {out_dir / 'test_scores.jsonl'}")
    print(f"wrote: {out_dir / 'calibration_metadata.json'}")
    print()

    print("Platt calibration:")
    print(f"  a:                         {a:.8f}")
    print(f"  b:                         {b:.8f}")
    print(f"  cosine threshold p=0.5:    {decision_cosine_threshold}")
    print()

    print("validation comparison:")
    print(f"  hard 0/1 overall:          {hard_metrics['overall']:.6f}")
    print(f"  platt overall:             {platt_metrics['overall']:.6f}")
    print(f"  platt+abstain overall:     {platt_abstain_metrics['overall']:.6f}")
    print()

    print("selected abstention:")
    print(f"  delta:                     {best_delta}")
    print(f"  validation abstain %:      {100.0 * metadata['validation_nonanswer_rate_selected']:.2f}%")
    print(f"  test abstain %:            {100.0 * metadata['test_nonanswer_rate']:.2f}%")
    print()

    print("full validation metrics:")
    print("  hard 0/1:")
    for k, v in hard_metrics.items():
        print(f"    {k}: {v:.6f}")

    print("  platt:")
    for k, v in platt_metrics.items():
        print(f"    {k}: {v:.6f}")

    print("  platt+abstain:")
    for k, v in platt_abstain_metrics.items():
        print(f"    {k}: {v:.6f}")


if __name__ == "__main__":
    main()
